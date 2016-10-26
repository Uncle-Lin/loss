#!/usr/bin/env python
# coding=utf-8

import tensorflow as tf

def Q_function1(logits,labels,L):
	'''
		logits:softmax output with size [batch_size,classes]
		labels:int value with size [batch_size]
		this function implement the following formulation:
		Q(x) = 1 - exp(-(c1-c2)^2)/L if |c1 - c2|<=L 
		Q(x) = 1 if |c1 - c2| > L
	'''
	labels = tf.cast(labels,tf.int64)
	real_labels = tf.arg_max(logits,1)
	sub_val = labels - real_labels
	bool_val = tf.abs(sub_val) <= L*tf.ones_like(sub_val)
	bool_val = tf.cast(bool_val,tf.float32)
	return 1.0 - bool_val*tf.exp(-sub_val*sub_val)

def Q_function2(logits,labels,L):
	'''
		logits:softmax output with size [batch_size,classes]
		labels:int value with size [batch_size]
		this function implement the following formulation:
		Q(x) = |c1 - c2|/L if |c1 - c2|<=L 
		Q(x) = 1 if |c1 - c2| > L
	'''	
	labels = tf.cast(labels,tf.int64)
	real_labels = tf.arg_max(logits,1)
	sub_val = labels - real_labels
	bool_val = tf.abs(sub_val) <= L*tf.ones_like(sub_val)
	bool_val = tf.cast(bool_val,tf.float32)
	return (1 - bool_val)*1 + bool_val*tf.abs(tf.cast(sub_val,tf.float32)/2.0


def Q_function2(logits,labels,L):
	'''
		logits:softmax output with size [batch_size,classes]
		labels:int value with size [batch_size]
		this function implement the following formulation:
		Q(x) = 0 if |c1 - c2|<=L 
		Q(x) = 1 if |c1 - c2| > L
	'''	
	labels = tf.cast(labels,tf.int64)
	real_labels = tf.arg_max(logits,1)
	sub_val = labels - real_labels
	bool_val = tf.abs(sub_val) <= L*tf.ones_like(sub_val)
	bool_val = tf.cast(bool_val,tf.float32)
	return (1-bool_val)*1 


def Q_function2(logits,labels,L):
	'''
		logits:softmax output with size [batch_size,classes]
		labels:int value with size [batch_size]
		this function implement the following formulation:
		Q(x) = |c1 - c2|/L 
	'''	
	labels = tf.cast(labels,tf.int64)
	real_labels = tf.arg_max(logits,1)
	sub_val = labels - real_labels
	return tf.abs(tf.cast(sub_val,tf.float32)/tf.cast(L,tf.float32)

