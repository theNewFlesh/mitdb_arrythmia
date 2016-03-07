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
from StringIO import StringIO
import re

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
# ------------------------------------------------------------------------------

def classification_report_to_dataframe(report):
	'''
	converts a classification_report to a DataFrame

	Args:
		report (str): output of classification_report

	Returns:
		DataFrame
	'''
	return pd.read_table(StringIO(re.sub('avg / total', 'avg/total' , report)), sep=' +', engine='python')

def prep_data(data, balance=True):
	'''
	prepares a machine learning dataframe from mitdb HDFStore object

	Args:
		data (HDFStore): mitdb HDFStore data
		balance (bool, opt): balance arrythmia/not arrythmia classes

	Returns:
		DataFrame
	'''
	records = filter(lambda x: re.search('record', x), data.keys())
	records = [data[key] for key in records]
	
	data = DataFrame()
	for record in records:
		if record.arrythmia.sum() > 1:
			data = pd.concat([data, conform_data(record)])

	data.reset_index(drop=True, inplace=True)
	
	if balance:
		mask = data.y == 1
		size = data[mask].shape[0]
		index = np.random.choice(data[~mask].index, size)
		index = np.concatenate([index, data[mask].index])
		data = data.ix[index]
		data.reset_index(drop=True, inplace=True)
		
	return data

def get_results(clf, params, x_train, y_train, x_val, y_val, cols):
	'''
		prints results of GridSearch training

		Args:
			clf (classifier): sklearn classifier to use
			params (dict): GridSearch params to use
			x_train (np.array): x training data
			y_train (np.array): y training labels
			x_val (np.array): x validation data
			y_val (np.array): y validation labels
			cols (list): column names

		Returns:
			grid (GridSearch)
	'''
	grid = GridSearchCV(clf, params, cv=5)
	grid.fit(x_train, y_train)
	pred = grid.best_estimator_.predict(x_val)
	print(classification_report(y_val, pred))
	if isinstance(clf, RandomForestClassifier): 
		importances = grid.best_estimator_.feature_importances_
		print(Series(importances, index=cols).sort_values(ascending=False).head(30))
	return grid

def small_train_size_test(clf, params, x_train, y_train, cols):
	'''
		runs multiple gridsearches on increasingly smaller train sizes and aggregaates the results

		Args:
			clf (classifier): sklearn classifier to use
			params (dict): GridSearch params to use
			x_train (np.array): x training data
			y_train (np.array): y training labels
			cols (list): column names

		Returns:
			reports (list of classification report DataFrames)
			importances (list of importances Series')
	'''
	reports = []
	importances = []
	for test_size in np.arange(0.98, 0.995, 0.001):
		x_train_, x_val, y_train_, y_val = train_test_split(x_train, y_train, test_size=test_size, random_state=np.random.randint(0, 100))
	
		grid = GridSearchCV(clf, params, cv=2)
		grid.fit(x_train_, y_train_)
		pred = grid.best_estimator_.predict(x_val)
		report = classification_report(y_val, pred)
		report = classification_report_to_dataframe(report)
		report['train_size'] = x_train_.shape[0]
		reports.append(report)
		if isinstance(clf, RandomForestClassifier): 
			imp = grid.best_estimator_.feature_importances_
			imp = Series(imp, index=cols).sort_values(ascending=False)
			importances.append(imp)
			
	return reports, importances
# ------------------------------------------------------------------------------

def main():
	pass
# ------------------------------------------------------------------------------

__all__ = [
	'classification_report_to_dataframe',
	'prep_data',
	'get_results',
	'small_train_size_test'
]

if __name__ == '__main__':
	help(main)