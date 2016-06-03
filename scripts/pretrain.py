#!/usr/bin/env python3
# code by Jon May [jonmay@isi.edu]. Port of code by Deniz Yuret with some interface improvements
import argparse
import sys
import codecs
if sys.version_info[0] == 2:
  from itertools import izip
else:
  izip = zip
from collections import defaultdict as dd
import re
import os.path
import os
import gzip
import tempfile
import shutil
import shlex
import atexit
import operator
from subprocess import check_call
scriptdir = os.path.dirname(os.path.abspath(__file__))

reader = codecs.getreader('utf8')
writer = codecs.getwriter('utf8')


def prepfile(fh, code):
  ret = gzip.open(fh.name, code) if fh.name.endswith(".gz") else fh
  if sys.version_info[0] == 2:
    if code.startswith('r'):
      ret = reader(fh)
    elif code.startswith('w'):
      ret = writer(fh)
    else:
      sys.stderr.write("I didn't understand code "+code+"\n")
      sys.exit(1)
  return ret

# grabbed from https://hg.python.org/cpython/file/default/Lib/argparse.py
# written by steven bethard! super cool!
class py34FileType(object):
  """Factory for creating file object types

  Instances of FileType are typically passed as type= arguments to the
  ArgumentParser add_argument() method.

  Keyword Arguments:
      - mode -- A string indicating how the file is to be opened. Accepts the
          same values as the builtin open() function.
      - bufsize -- The file's desired buffer size. Accepts the same values as
          the builtin open() function.
      - encoding -- The file's encoding. Accepts the same values as the
          builtin open() function.
      - errors -- A string indicating how encoding and decoding errors are to
          be handled. Accepts the same value as the builtin open() function.
  """

  def __init__(self, mode='r', bufsize=-1, encoding=None, errors=None):
    self._mode = mode
    self._bufsize = bufsize
    self._encoding = encoding
    self._errors = errors

  def __call__(self, string):
    # the special argument "-" means sys.std{in,out}
    if string == '-':
      if 'r' in self._mode:
        return _sys.stdin
      elif 'w' in self._mode:
        return _sys.stdout
      else:
        msg = _('argument "-" with mode %r') % self._mode
        raise ValueError(msg)

      # all other arguments are used as file names
    try:
      return open(string, self._mode, self._bufsize, self._encoding,
                  self._errors)
    except OSError as e:
      message = _("can't open '%s': %s")
      raise ArgumentTypeError(message % (string, e))

  def __repr__(self):
    args = self._mode, self._bufsize
    kwargs = [('encoding', self._encoding), ('errors', self._errors)]
    args_str = ', '.join([repr(arg) for arg in args if arg != -1] +
                         ['%s=%r' % (kw, arg) for kw, arg in kwargs
                          if arg is not None])
    return '%s(%s)' % (type(self).__name__, args_str)


def main():
  parser = argparse.ArgumentParser(description="python port of yuret/zoph code. train a model initialized with a parent model's params",
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("--parent", "-p", nargs='?', type=py34FileType('r', encoding="utf-8"), default=sys.stdin, help="parent model file")
  parser.add_argument("--trainsource", "-ts", nargs='?', type=py34FileType('r', encoding="utf-8"), default=sys.stdin, help="train source file")
  parser.add_argument("--traintarget", "-tt", nargs='?', type=py34FileType('r', encoding="utf-8"), default=sys.stdin, help="train target file")
  parser.add_argument("--devsource", "-ds", help="dev source file")
  parser.add_argument("--devtarget", "-dt", help="dev target file")
  parser.add_argument("--rnnbinary", default=os.path.join(scriptdir, 'ZOPH_RNN'), help="zoph rnn nmt binary")
  parser.add_argument("--child", "-c",  help="output child model file")
  parser.add_argument("--dropout", "-d", type=float, default=0.5, help="dropout rate (1 = always keep)")
  parser.add_argument("--learning_rate", "-l", type=float, default=0.5, help="learning rate")
  parser.add_argument("--adaptive_decrease_factor", "-A", type=float, default=0.9, help="adaptive decrease factor")
  parser.add_argument("--parameter_range", "-P", type=float, default=0.05, help="initial randomly assigned range (centered on 0)")
  parser.add_argument("--whole_clip_gradients", "-w", type=float, default=5, help="clip gradients if they exceed (?) this")
  parser.add_argument("--longest_sent", "-L", type=int, default=100, help="longest sentence; longer ones are discarded")
  parser.add_argument("--minibatch_size", "-m", type=int, default=128, help="items per minibatch")
  parser.add_argument("--number_epochs", "-n", type=int, default=100, help="training epochs")
  parser.add_argument("--attention_model", type=bool, default=True, help="use attention model")
  parser.add_argument("--feed_input", type=bool, default=True, help="use feed input") 
  parser.add_argument("--train_source_input_embedding", type=bool, default=True)
  parser.add_argument("--train_target_input_embedding", type=bool, default=False)
  parser.add_argument("--train_target_output_embedding", type=bool, default=False)
  parser.add_argument("--train_source_RNN", type=bool, default=True)
  parser.add_argument("--train_target_RNN", type=bool, default=True)
  parser.add_argument("--train_attention_target_RNN", type=bool, default=True)
  parser.add_argument("--logfile", default="./log", help="where to pipe stdout of rnn binary")
  parser.add_argument("--other_rnn_arguments", default="", help="other arguments to pass to RNN. fully formed, quoted string")
  parser.add_argument("--cuda_lib_string", default="/home/nlg-05/zoph/cudnn_v4/lib64/:/usr/usc/cuda/7.0/lib64", help="cuda libraries that must be added to LD_LIBRARY_PATH")

  # TODO: options; as string or separately?

  workdir = tempfile.mkdtemp(prefix=os.path.basename(__file__), dir=os.getenv('TMPDIR', '/tmp'))

  def cleanwork():
    shutil.rmtree(workdir, ignore_errors=True)
  atexit.register(cleanwork)


  try:
    args = parser.parse_args()
  except IOError as msg:
    parser.error(str(msg))

  parent = prepfile(args.parent, 'r')
  trainsource = prepfile(args.trainsource, 'r')
  traintarget = prepfile(args.traintarget, 'r')
  prechild = prepfile(open(args.child+".last", 'w', encoding="utf-8"), 'w')


  # get train source vocab, sorted lexicographically
  vocab = dd(int)
  for line in trainsource:
    for tok in line.strip().split():
      vocab[tok]+=1
  # vocabulary sorted by frequency, most frequent first
  vocab = list(map (lambda x: x[0], sorted(vocab.items(), key=operator.itemgetter(1), reverse=True)))

  print("Vocab length %d" % len(vocab))
  # get layer info from parent model
  line = parent.readline()
  (nlayer, nhidden, ntarget, nsource, *_) = map(int, line.strip().split())
  prechild.write(line)
  if len(vocab) < nsource:
    sys.stderr.write("Error: child vocabulary (%d) larger than parent vocabulary (%d)\n" % (len(vocab), nsource))
    sys.exit(1)
  #moption = "-M "+"0 "*(nlayer-1)+"1 1"
  moption=""
  line = parent.readline()
  if re.match("^=+$", line) is None:
    sys.stderr.write("Error: unexpected line "+line)
    sys.exit(1)
  prechild.write(line)

  # replace parent vocab with child vocab
  line = parent.readline()
  if re.match("^0 <UNK>$", line) is None:
    sys.stderr.write("Error: unexpected line "+line)
    sys.exit(1)
  prechild.write(line)
  for line in parent:
    if re.match("^=+$", line) is not None:
      prechild.write(line)
      break
    i, _ = line.strip().split()
    prechild.write("%s %s\n" % (i, vocab[int(i)-1]))
  # replace rest of parent model
  for line in parent:
    prechild.write(line)
  prechild.close()
  parent.close()
  # launch training

  maincmd = "%s -C %s %s %s -B %s -a %s %s %s" % (args.rnnbinary, args.trainsource.name, args.traintarget.name, prechild.name, args.child, args.devsource, args.devtarget, moption)
  cmdargs = "--dropout %f --learning-rate %f --adaptive-decrease-factor %f --parameter-range %f %f --whole-clip-gradients %f --longest-sent %d --minibatch-size %d --attention-model %d --number-epochs %d --feed-input %d --train-source-input-embedding %d --train-target-input-embedding %d --train-target-output-embedding %d --train-source-RNN %d --train-target-RNN %d --train-attention-target-RNN %d --logfile %s --tmp-dir-location %s" % (args.dropout, args.learning_rate, args.adaptive_decrease_factor, -args.parameter_range, args.parameter_range, args.whole_clip_gradients, args.longest_sent, args.minibatch_size, args.attention_model, args.number_epochs, args.feed_input, args.train_source_input_embedding, args.train_target_input_embedding, args.train_target_output_embedding, args.train_source_RNN, args.train_target_RNN, args.train_attention_target_RNN, args.logfile, workdir)
  cmd="%s %s %s" % (maincmd, cmdargs, args.other_rnn_arguments)
  sys.stderr.write("Executing "+cmd+"\n")
  cmdlist = shlex.split(cmd)
  sys.exit(check_call(cmdlist, env=dict(os.environ, **{"LD_LIBRARY_PATH":"%s:%s" % (os.environ["LD_LIBRARY_PATH"], args.cuda_lib_string)})))

if __name__ == '__main__':
  main()
