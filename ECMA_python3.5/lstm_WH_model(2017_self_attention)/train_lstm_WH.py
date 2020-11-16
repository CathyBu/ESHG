# coding=utf-8
import re

import numpy as np
import tensorflow as tf
import tflearn
from sklearn.utils import shuffle

from lstm_WH_model.model_lstm_WH import SelfAttentive
from reader import load_csv, VocabDict
import diceEval
import data_process
'''
parse
'''

tf.app.flags.DEFINE_integer('num_epochs',5, 'number of epochs to train')
tf.app.flags.DEFINE_integer('batch_size', 1, 'batch size to train in one step')
# tf.app.flags.DEFINE_integer('labels', 5, 'number of label classes')
tf.app.flags.DEFINE_integer('labels', 2, 'number of label classes')
# tf.app.flags.DEFINE_integer('word_pad_length', 60, 'word pad length for training')
tf.app.flags.DEFINE_integer('word_pad_length', 20, 'word pad length for training')
tf.app.flags.DEFINE_integer('decay_step', 100, 'decay steps')
tf.app.flags.DEFINE_float('learn_rate', 1e-2, 'learn rate for training optimization')
tf.app.flags.DEFINE_boolean('shuffle', True, 'shuffle data FLAG')
tf.app.flags.DEFINE_boolean('train', True, 'train mode FLAG')
tf.app.flags.DEFINE_boolean('visualize', True, 'visualize FLAG')
tf.app.flags.DEFINE_boolean('penalization', False, 'penalization FLAG')

FLAGS = tf.app.flags.FLAGS

num_epochs = FLAGS.num_epochs
batch_size = FLAGS.batch_size
tag_size = FLAGS.labels
word_pad_length = FLAGS.word_pad_length
lr = FLAGS.learn_rate

TOKENIZER_RE = re.compile(r"[A-Z]{2,}(?![a-z])|[A-Z][a-z]+(?=[A-Z])|[\'\w\-]+", re.UNICODE)
def token_parse(iterator):
  for value in iterator:
    return TOKENIZER_RE.findall(value)

tokenizer = tflearn.data_utils.VocabularyProcessor(word_pad_length, tokenizer_fn=lambda tokens: [token_parse(x) for x in tokens])
label_dict = VocabDict()

def string_parser(arr, fit):
  if fit == False:
    return list(tokenizer.transform(arr))
  else:
    return list(tokenizer.fit_transform(arr))

# preName='QRJD' #69
# preName = 'HXHY' #17
# preName = 'HT' #137
# preName = 'ZJG' #17
# preName = 'ZX' #11
# preName = 'XZ' #5
# preName = 'LS' #9
# preName = 'MM'  #11
# preName = 'JP' #124
# preName = 'ZK' #27
# preName = 'AS' #55
# preName = 'HXZT' #1
# preName = 'ZY' #6
preName = 'TL' #13
evalNum =13

model = SelfAttentive()
with tf.Session() as sess:
  # build graph
  model.build_graph(n=word_pad_length)
  # Downstream Application
  with tf.variable_scope('DownstreamApplication'):
    global_step = tf.Variable(0, trainable=False, name='global_step')
    learn_rate = tf.train.exponential_decay(lr, global_step, FLAGS.decay_step, 0.95, staircase=True)
    labels = tf.placeholder('float32', shape=[None, tag_size])
    net = tflearn.fully_connected(model.M, 50, activation='relu')
    logits = tflearn.fully_connected(net, tag_size, activation=None)
    loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits), axis=1)
    if FLAGS.penalization == True:
      p_coef = 0.004
      p_loss = p_coef * model.P
      loss = loss + p_loss
      p_loss = tf.reduce_mean(p_loss)
    loss = tf.reduce_mean(loss)
    params = tf.trainable_variables()
    #clipped_gradients = [tf.clip_by_value(x, -0.5, 0.5) for x in gradients]
    optimizer = tf.train.AdamOptimizer(learn_rate)
    grad_and_vars = tf.gradients(loss, params)
    clipped_gradients, _ = tf.clip_by_global_norm(grad_and_vars, 0.5)
    opt = optimizer.apply_gradients(zip(clipped_gradients, params), global_step=global_step)

  # Start Training
  sess.run(tf.global_variables_initializer())

  words, tags = load_csv('../data/trainTCM/TCM_train_%s.csv'%preName, target_columns=[0], columns_to_ignore=None,
                         target_dict=label_dict)
  words = string_parser(words, fit=True)
  if FLAGS.shuffle == True:
    words, tags = shuffle(words, tags)
  word_input = tflearn.data_utils.pad_sequences(words, maxlen=word_pad_length)
  total = len(word_input)
  step_print = int((total/batch_size) / 20)

  if FLAGS.train == True:
    print('start training')
    for epoch_num in range(num_epochs):
      epoch_loss = 0
      step_loss = 0
      for i in range(int(total/batch_size)):
        batch_input, batch_tags = (word_input[i*batch_size:(i+1)*batch_size], tags[i*batch_size:(i+1)*batch_size])
        # print('zzzzzzz',batch_tags)
        train_ops = [opt, loss, learn_rate, global_step]
        if FLAGS.penalization == True:
          train_ops += [p_loss]
        result = sess.run(train_ops, feed_dict={model.input_pl: batch_input, labels: batch_tags})
        step_loss += result[1]
        epoch_loss += result[1]
        if i % step_print == (step_print-step_print):
          if FLAGS.penalization == True:
            print('step_log: (epoch: {%s}, step: {%s}, global_step: {%s}, learn_rate: {%s}), Loss: {%s}, Penalization: {%s})'%(epoch_num,i,result[3],result[2],(step_loss/step_print),result[4]))
          else:
            print('step_log: (epoch: {%s}, step: {%s}, global_step: {%s}, learn_rate: {%s}), Loss: {%s})'%(epoch_num,i,result[3],result[2],(step_loss/step_print)))
          #print(f'{result[4]}')
          step_loss = 0
      print('***')
      print('epoch {%s}: (global_step: {%s}), Average Loss: {%s})'%(epoch_num,result[3],(epoch_loss/(total/batch_size))))
      print('***\n')
    saver = tf.train.Saver()
    saver.save(sess, './WH_model/0330_model_Lstm_WH_r1_%s_epoches%s.ckpt'%(preName,FLAGS.num_epochs))
  else:
    saver = tf.train.Saver()
    saver.restore(sess, './WH_model/0330_model_Lstm_WH_r1_%s_epoches%s.ckpt'%(preName,FLAGS.num_epochs))

  allDice=[]
  evalCount = 0
  print('start testing')
  words, tags = load_csv('../data/testData/TCM_testEval3_%s.csv'%preName, target_columns=[0], columns_to_ignore=None, target_dict=label_dict)
  words_with_index = string_parser(words, fit=True)
  word_input = tflearn.data_utils.pad_sequences(words_with_index, maxlen=word_pad_length)
  total = len(word_input)
  rs = 0.
  #load evalData start
  evalData=[]
  evalCav='../data/evalData/%s_evaluate.csv'%preName
  evalList=data_process.read_csv(evalCav)
  for item in evalList:
    evalData.append(item)
  # load evalData end

  if FLAGS.visualize == True:
    f = open('./0330HTML/visualizeTCM_%s_LSTM_WH_epoches%s_r1.html'%(preName,FLAGS.num_epochs), 'w')
    # f = open('./html_WH/dp_visualizeTCM_%s_LSTM_WH_epoches%s_r1.html'%(preName,FLAGS.num_epochs), 'w')
    f.write('<html style="margin:0;padding:0;"><meta http-equiv="Content-Type" content="text/html; charset=UTF-8"><body style="margin:0;padding:0;">\n')

  for i in range(int(total/batch_size)):
    batch_input, batch_tags = (word_input[i*batch_size:(i+1)*batch_size], tags[i*batch_size:(i+1)*batch_size])
    result = sess.run([logits, model.A], feed_dict={model.input_pl: batch_input, labels: batch_tags})
    #arr保存预测概率
    arr = result[0]
    if not np.argmax(arr[0]):
        preClass=True
    else:
        preClass = False
    for j in range(len(batch_tags)):
      if np.argmax(batch_tags[j])==0:
          if np.argmax(arr[j]) == np.argmax(batch_tags[j]):
            evalCount+=1

      rs+=np.sum(np.argmax(arr[j]) == np.argmax(batch_tags[j]))
    medicalList = []
    if FLAGS.visualize == True:
      f.write('<div style="margin:15px;">\n')
      #result[1][0]保存的是方剂中每个药物对应的attention因子，具体result[1][0][k][j]取出
      for k in range(len(result[1][0])):
        # f.write('\t<p> —— 测试方剂 %s (类标：%s ; 预测类标：%s)：—— </p>\n'%(i, tags[i],preClass))
        f.write('<p style="margin:10px;font-family:SimHei">\n')
        ww = TOKENIZER_RE.findall(words[i*batch_size][0])
        for j in range(word_pad_length):
          if result[1][0][k][j] <0.05:
            result[1][0][k][j]=0
          alpha = "{:.2f}".format(result[1][0][k][j])
          # print('注意力因子值：{:.2f}'.format(result[1][0][k][j]))
          if len(ww) <= j:
            w = "   "
          else:
            w = ww[j]
            if result[1][0][k][j] >= 0.05:
              medicalList.append(w)
          f.write('\t<span style="margin-left:3px;background-color:rgba(255,0,0,%s)">%s</span>\n' % (alpha, w))
        f.write('</p>\n')
        if i < evalNum:
          if preClass == True:
            # print('配伍评估药组：', medicalList)
            allDice.append(diceEval.evalMedicalDice(medicalList, evalData))
          else:
            allDice.append(0)
        # if i < evalNum:
          # f.write('\t<b>配伍评估药组： %s ,dice = %s</b>\n' % (','.join(medicalList), allDice[i]))
      f.write('</div>\n')

  if FLAGS.visualize == True:
    f.write('\t<p>Test accuracy: %s</p>\n' % (rs / total))
    f.write('\t<p>该功效下%s个经典方剂(即测试集前%s个方剂) accuracy ：%s</p>\n' % (evalNum,evalNum,evalCount / evalNum))
    f.write('\t<p>该功效下%s个经典方剂 avg-dice : %s</p>\n' % (evalNum,sum(allDice) / evalNum))
    f.write('</body></html>')
    f.close()

  print('Test accuracy(all test data): %s' % (rs / total))
  print('该功效下%s个经典方剂(即测试集前%s个方剂)的accuracy评估：%s' % (evalNum,evalNum,evalCount / evalNum))
  print('allDice,evalCount', allDice, evalCount)
  sumValue = 0
  for i in allDice:
    sumValue += i
  print('avg-dice:%s' % (sumValue / evalNum))
  sess.close()
