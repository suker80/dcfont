import argparse
import os
from model import MCGAN
import tensorflow.contrib as gan
parser = argparse.ArgumentParser()

parser.add_argument("--lr",type=float,help="learning_rate",default=0.0002)

parser.add_argument("--batch_size",type=int,help="batch size",default=15)

parser.add_argument("--input_size",type=int,help="image input size ",default=64)

parser.add_argument("--output_size",type=int,help="image output size",default=64)

parser.add_argument("--epoch",type=int,help="number of epochs",default=1000)

parser.add_argument("--step",type=int,help="how many roop in a epoch",default=200)

parser.add_argument("--mode",type=str,help="select mode training or test",default='test')

parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='./checkpoint/', help='models are saved here')

parser.add_argument('--train_root',type=str,default='kor_dataset')

parser.add_argument('--niter',type=float,default=500)
parser.add_argument('--niter_decay',type=float,default=100)


args=parser.parse_args()


if not os.path.exists(args.checkpoint_dir):
    os.makedirs(args.checkpoint_dir)

model=MCGAN(output_size=args.output_size,
            input_size=args.input_size,
            epoch=args.epoch,
            step=args.step,
            lr=args.lr,
            batch_size=args.batch_size,
            checkpoint_dir=args.checkpoint_dir,
            mode=args.mode
            ,niter=args.niter,niter_decay=args.niter_decay
            )


if args.mode == 'train':
    model.train(args.train_root)
elif args.mode =='test':
    model.test('checkpoint/kor_model-950')
