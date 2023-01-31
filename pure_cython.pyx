#!/usr/bin/env python
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# coding: utf-8

import cython
import numpy as np
cimport numpy as np

from libc.math cimport exp
from libc.string cimport memset

from cpython cimport PyCapsule_GetPointer
import scipy.linalg.blas as cblas


# y += alpha * x
ctypedef void (*saxpy_ptr) (const int *N, const float *alpha, const float *X, const int *incX, float *Y, const int *incY) nogil

# dot(x, y); return value should be `float`, but it only works with `double` (?!)
ctypedef double (*sdot_ptr) (const int *N, const float *X, const int *incX, const float *Y, const int *incY) nogil



cdef saxpy_ptr saxpy=<saxpy_ptr>PyCapsule_GetPointer(cblas.saxpy._cpointer, NULL)
cdef sdot_ptr sdot=<sdot_ptr>PyCapsule_GetPointer(cblas.sdot._cpointer, NULL)

REAL = np.float32
ctypedef np.float32_t REAL_t



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline void fast_sentence(
    np.uint32_t[::1] word_nodes, # indexes of nodes in path of huffman tree
    np.uint8_t[::1] word_code, # 0 and 1 (left, right directions)
    unsigned long int codelen, # length of path to the word in huffman tree
    REAL_t[:, ::1] _w0,  # result embedding matrix
    REAL_t[:, ::1] _w1, # context (huffman tree nodes) embedding matrix
    int size, # hidden size of matrix
    np.uint32_t context_word_index, # index of context word
    REAL_t alpha, # gradient descent learning rate
    REAL_t[::1] _work) nogil:
    cdef long long i, j # indexes to perform vector/matrix runs
    cdef long long row_w0 = context_word_index*size, row_w1 # row index in w0 and w1 matrix
    cdef REAL_t f, g # variables to accumulate temp result
    cdef REAL_t *work = &_work[0] # array pointer to accumulate vector temp result
    cdef REAL_t *w0 = &_w0[0, 0] # pointer to start of w0
    cdef REAL_t *w1 = &_w1[0, 0]# pointer to start of w1
    memset(work, 0, size * cython.sizeof(REAL_t))
    for i in range(0, codelen):
        f = <REAL_t>0.0
        row_w1 = <long long >word_nodes[i] * <long long>size
        for j in range(0, size):
            f += w0[row_w0 + j] * w1[row_w1 + j]
        f = 1 / (1 + exp(-f))
        g = (1 - word_code[i] - f) * alpha
        for j in range(0, size):
            work[j] += g * w1[row_w1 + j]
        for j in range(0, size):
            w1[row_w1+j] += g * w0[row_w0+ j]
    for i in range(0, size):
        w0[row_w0+i] += <REAL_t>work[i]








    



def train_sentence(model, sentence, alpha):
    """
    sentence is a list of objects
    word.nodes - np.array of indexes of nodes in path of huffman tree
    word.code - np.array of 0 and 1 (left, right directions)
    """
    work = np.zeros(model.hidden_size, dtype=REAL)  # each thread must have its own work memory
    reduced_window = 0
    for word_pos, word in enumerate(sentence):
        if word is None:
            continue  # OOV word in the input sentence => skip
        # now go over all words from the window, predicting each one in turn
        start = max(0, word_pos - model.window + reduced_window)
        for context_pos, context_word in enumerate(sentence[start : word_pos + model.window + 1], start):
            if context_pos == word_pos or context_word is None:
                # don't train on OOV words and on the `word` itself
                continue
            fast_sentence(word.point, word.code, word.codelen, model.w0, model.w1, model.hidden_size,
                          context_word.index, alpha, work)