# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Test for version 1 of the zero_out op."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import  sys
import tensorflow as tf
#from tensorflow.examples.adding_an_op import cuda_op
import bf16cut_fp_op


with tf.Session(''):
  rand1  = tf.random.uniform([2,2,2],-100,100,dtype=tf.dtypes.float32)
  rand1_prt = tf.print("rand1_prt : ", rand1,output_stream=sys.stdout)
  with tf.control_dependencies([rand1_prt]):
    result = bf16cut_fp_op.bf16cut_fp(rand1)
    result_prt = tf.print("result_prt : ", result,output_stream=sys.stdout)
    with tf.control_dependencies([result_prt]):
      #result_id = bf16cut_fp_op.bf16cut_fp(result)
      diffv = tf.math.subtract(result , rand1)
      absv  = tf.math.abs(diffv)
      absr  = tf.math.abs(rand1)
      result_id = tf.math.divide(absv,absr)
      print("final step : "+str(result_id.eval()))

