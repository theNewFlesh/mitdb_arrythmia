#! /usr/bin/env python
# Alex Braun 03.07.2016

# ------------------------------------------------------------------------------
# The MIT License (MIT)

# Copyright (c) 2016 Alex Braun

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
# ------------------------------------------------------------------------------

from __future__ import print_function, with_statement
import os
import re
from subprocess import Popen, PIPE
from datetime import datetime
import requests
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
# ------------------------------------------------------------------------------

def get_raw_data(pardir):
	'''
	scrapes physionet.org for mitdb data

	Args:
		pardir (str): fullpath of parent directory of raw files

	Returns:
		None
	'''
	url = 'https://www.physionet.org/physiobank/database/mitdb/'
	html = requests.get(url).content
	soup = BeautifulSoup(html)
			
	links = [x['href'] for x in soup.select('a')]
	links = filter(lambda x: re.search('\.atr|\.dat|\.hea', x, re.I), links)
	links = sorted(links)
	
	targets = [os.path.join(pardir, x) for x in links]
	links = [url + x for x in links]

	for link, target in zip(links, targets):
		with open(target, 'w+') as f:
			f.write(requests.get(link).content)

def extract_records(source, target):
	'''
	extracts csv data and txt annotations from raw files in source directory
	REQUIRES: wfdb to be installed  in order to run rdsamp and rdann
			  https://physionet.org/physiotools/wfdb.shtml

	Args:
		source (str): fullpath of mitdb directory
		target (str): fullpath of target directory

	Returns:
		None
	'''
	os.chdir(os.path.split(source)[0])
	records = set()
	for item in os.listdir(source):
		found = re.search('^(\d\d\d)\.', item)
		if found:
			records.add(found.group(1))

	csv_cmd = 'rdsamp -r mitdb/{rec} -c -v -pe > '   + target + '/mitdb.{rec}.csv'
	ann_cmd = 'rdann -r mitdb/{rec} -a atr -v -e > ' + target + '/mitdb.ann.{rec}.txt'
	for record in records:
		Popen(csv_cmd.format(rec=record), shell=True, stdout=PIPE)
		Popen(ann_cmd.format(rec=record), shell=True, stdout=PIPE)

def read_csv(filepath):
	'''
	ETLs a mitdb csv file

	Args:
		filepath (str): fullpath of single mitdb csv file

	Returns:
		DataFrame
	'''
	data = pd.read_csv(filepath)
	cols = ['elapsed_time', 'mlii_mV', 'v5_mV']
	data.columns = cols
	data.reset_index(drop=True, inplace=True)
	data = data.ix[1:]
	data.reset_index(drop=True, inplace=True)
	
	index = data.elapsed_time.apply(lambda x: x[1:-1])
	
	zero = datetime.strptime('00:00.00', '%M:%S.%f')
	index = data.elapsed_time.apply(lambda x: datetime.strptime(x[1:-1], '%M:%S.%f') - zero)
	data.index = index
	data.drop('elapsed_time', axis=1, inplace=True)
	
	data.mlii_mV = data.mlii_mV.astype(float)
	data.v5_mV = data.v5_mV.astype(float)
	
	return data

def read_ann(filepath):
	'''
	reads mitdb txt annotation file

	Args:
		filepath (str): fullpath of mitdb txt annotation file

	Returns:
		DataFrame
	'''
	data = pd.read_table(filepath, sep='\s\s+|\t| C')
	cols = ['elapsed_time', 'sample_num', 'type', 'sub', 'chan', 'num', 'aux']
	data.columns = cols
	
	zero = datetime.strptime('00:00.00', '%M:%S.%f')
	index = data.elapsed_time.apply(lambda x: datetime.strptime(x, '%M:%S.%f') - zero)
	data.index = index
	data.drop('elapsed_time', axis=1, inplace=True)
	
	data.sample_num = data.sample_num.astype(int)
	data['sub'] = data['sub'].astype(int)
	data.chan = data.chan.astype(int)
	data.num = data.num.astype(int)
	
	return data

def compile_data(csv, ann):
	'''
	compiles mitdb csv and annotation files into singular DataFrame

	Args:
		csv (str): fullpath of mitdb csv file
		ann (str): fullpath of mitdb txt annotation file

	Returns:
		DataFrame
	'''
	a = read_csv(csv)
	b = read_ann(ann)
	data = pd.concat([a,b], axis=1)
	
	info = get_info()
	symbols = info[info.arrythmia].symbol.tolist()
	y = data.type.apply(lambda x: x in symbols).astype(int)
	y += data.aux.apply(lambda x: x in symbols).astype(int)
	y = y.astype(bool).astype(int)
	data['arrythmia'] = y
	
	symbols = ['N', 'L', 'R']
	y = data.type.apply(lambda x: x in symbols).astype(int)
	y += data.aux.apply(lambda x: x in symbols).astype(int)
	y = y.astype(bool).astype(int)
	data['normal'] = y
	
	return data

def write_data(source, target):
	'''
	writes all mitdb source directory csvs and txt annotations to a single hdf5 file

	Args:
		source (str): full path of directory containing mitdb csvs and txt annotations
		target (str): fullpath of target file

	Returns:
		None
	'''
	csvs = filter(lambda x: True if re.search('csv$', x) else False, os.listdir(source))
	csvs = [os.path.join(source, csv) for csv in csvs]
	csvs = sorted(csvs)
	
	anns = filter(lambda x: True if re.search('ann', x) else False, os.listdir(source))
	anns = [os.path.join(source, ann) for ann in anns]
	anns = sorted(anns)

	nums = [re.search('(\d\d\d)', x).group(1) for x in csvs]
	
	for num, csv, ann in zip(nums, csvs, anns):
		datum = compile_data(csv, ann)
		datum.to_hdf(target, 'record_' + num)
	
	info = get_info()
	info.to_hdf(target, 'info')
		
def has_word(item, words):
	'''determines if a word is in words'''
	for word in words:
		if word.lower() in item.lower():
			return True
	return False

def get_info():
	'''
	scrapes physionet.org for mitdb metadata

	Args:
		None

	Returns:
		DataFrame
	'''
	url = 'https://www.physionet.org/physiobank/database/html/mitdbdir/intro.htm#annotations'
	html = requests.get(url).content
	soup = BeautifulSoup(html)
	info = soup.select('table')[-1]
	info = pd.read_html(str(info), header=0)[0]
	info.rename_axis(lambda x: x.lower(), axis=1, inplace=True)
	mask = info.symbol.apply(lambda x: True) 
	mask.ix[[20,36]] = False
	info = info[mask]
	info.reset_index(drop=True, inplace=True)
	
	lut = {
	'arrythmia': ['flutter', 'bigeminy', 'tachycardia'],
	'other':     ['bradycardia', 'abberated', 'premature', 'escape']
	}
	
	info['arrythmia'] = info.meaning.apply(lambda x: has_word(x, lut['arrythmia']))
	info['other'] = info.meaning.apply(lambda x: has_word(x, lut['other']))
	info.loc[0, 'symbol'] = 'N'
	return info

def get_delta(timestring=None, hours=0, minutes=0, seconds=0, microseconds=0):
	'''
	produces a time delta the size of the input

	Args:
		timestring (str, opt): string in 00:00:00.00 format
		hours (int, opt): number of hours
		minutes (int, opt): number of minutes
		seconds (int, opt): number of seconds
		mmicroseconds (int, opt): number of mmicroseconds

	Returns:
		timedelta
	'''
	if not timestring:
		timestring = ':'.join(map(str, [hours, minutes, seconds])) + '.' + str(microseconds)
	return datetime.strptime(timestring, '%H:%M:%S.%f') - datetime.strptime('0:0:0.0', '%H:%M:%S.%f')

def get_heartbeat(data, col):
	'''
	featurizes arrythmia data to indicate individual hearbeats

	Args:
		data (DataFrame): mitdb DataFrame
		col (str): column of mitdb DataFrame to base heartbeat feature on

	Returns:
		heartbeats (list): temporal list of heartbeat probabilities
	'''
	x1 = data.index.astype(int).tolist()
	y1 = data[col]
	y2 = pd.rolling_kurt(y1, 100)
	y3 = pd.rolling_std(y1 - pd.rolling_mean(y1, 10), 10)
	return reduce( lambda x,y: x*y, [y1, y2, y3] )

def conform_data(data):
	'''
	conforms mtdb data for machine learning purposes

	Args:
		data (DataFrame): mitdb data

	Returns:
		DataFrame
	'''
	def get_samples(index, y):
		delta_a = get_delta(microseconds=400000)
		delta_b = get_delta(microseconds=10000)
		a = index - delta_a
		b = index + delta_b

		intervals = zip(a,b)
		output = []

		for a,b in intervals[1:]:
			sample = data.loc[a:b, ['mlii_mV', 'v5_mV']]
			mlii_heartbeat = get_heartbeat(sample, 'mlii_mV')
			v5_heartbeat = get_heartbeat(sample, 'v5_mV')
			sample = Series(sample.as_matrix().ravel())
			sample['mlii_heartbeat_max'] = mlii_heartbeat.max()
			sample['mlii_heartbeat_var'] = mlii_heartbeat.var()
			sample['v5_heartbeat_max'] = v5_heartbeat.max()
			sample['v5_heartbeat_var'] = v5_heartbeat.var()
			sample['y'] = y
			
			output.append(sample)
		
		output = DataFrame(output)
		return output
	
	output = get_samples(data[data.normal == 1].index, 0)
	arr    = get_samples(data[data.arrythmia == 1].index, 1)
	if arr.shape[0] > 0:
		output = pd.concat([output, arr])
	return output
# ------------------------------------------------------------------------------

def main():
	pass
# ------------------------------------------------------------------------------

__all__ = [
	'get_raw_data',
	'extract_records',
	'read_csv',
	'read_ann',
	'compile_data',
	'write_data',
	'has_word',
	'get_info',
	'get_delta',
	'get_heartbeat',
	'conform_data'
]

if __name__ == '__main__':
	help(main)