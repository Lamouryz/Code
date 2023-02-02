import argparse
import os

def argparser():
  parser = argparse.ArgumentParser()
  # for model
  parser.add_argument(
      '--seq_window_lengths',
      type=int,
      nargs='+',
      # default = [4, 9, 12],
      default = [5],
  )
  parser.add_argument(
      '--smi_window_lengths',
      type=int,
      nargs='+',
      # default=[4, 8, 12],
  )
  parser.add_argument(
      '--num_windows',
      type=int,
      nargs='+',
      # default=[100, 200, 100],
  )

  parser.add_argument(
      '--max_seq_len',
      type=int,
      #default=(1000,1200),
  )
  parser.add_argument(
      '--max_smi_len',
      type=int,
       default=100,
      #default=(100,85),
  )

  parser.add_argument(
      '--num_epoch',
      type=int,
      default=100,
  )
  parser.add_argument(
      '--batch_size',
      type=int,
      default=128,
  )
  parser.add_argument(
      '--dataset_path',
      type=str,
      default='./data/kiba/',
  )
  parser.add_argument(
      '--problem_type',
      type=int,
      default=2,
  )

  parser.add_argument(
      '--is_log',
      type=int,
      default=0,
  )
  parser.add_argument(
      '--checkpoint_path',
      type=str,
      default='./checkpoints/',
  )
  parser.add_argument(
      '--log_dir',
      type=str,
      default='./tmp/',
  )
  parser.add_argument(
      '--lamda',
      type=int,
      default=[-3, -5],
      nargs='+',
  )

  FLAGS, unparsed = parser.parse_known_args()
  return FLAGS

def logging(msg, FLAGS):
  fpath = os.path.join( FLAGS.log_dir, "log.txt" )
  with open( fpath, "a" ) as fw:
    fw.write("%s\n" % msg)

