from src.app import run
import argparse

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main():
    parser = argparse.ArgumentParser(description='Run a whole lotta tests.')
    parser.add_argument('-depth', default=6, type=int, help='Depth of network (valid options are 6 or 12.)')
    parser.add_argument('-aug', default=True, type=str2bool, action='store', required=False, help='Whether or not to use data augmentation.')
    parser.add_argument('-mparams', default=True, type=str2bool, action='store', required=False, help='Whether or not to use Michael Parameters.')
    parser.add_argument('-position', default='None', type=str, help='The position of the single sparse layer. Default, no sparse layer.')
    parser.add_argument('-fsmult', default=1, type=int, help="Multiple of the output dimensionality that the filterset size should be.")
    parser.add_argument('-kdiv', default=1, type=int, help="Divisor of the output dimensionality which determines the number of nonzero values in the TopK operation.")
    parser.add_argument('-auxweight', default=0.5, type=float, help="Amount of weight to put on the auxiliary loss.")
    parser.add_argument('-load', default=True, type=str2bool, action='store', required=False, help="Whether or not to try and load latest model.")
    parser.add_argument('-usecase', default="regularize", type=str, help="random/pretrain/regularize/supervise use case selection.")
    parser.add_argument('-prefix', default="", type=str, required=False, help="save directory prefix")
    parser.add_argument('-lr', default=-99.0, type=float, required=False, help="starting learning rate")

    args = parser.parse_args()

    run(lr=args.lr, depth=args.depth, augmentation=args.aug, mparams=args.mparams, position=args.position, fsmult=args.fsmult, kdiv=args.kdiv, auxweight=args.auxweight, loadmodel=args.load, usecase=args.usecase, prefix=args.prefix)

if __name__ == "__main__":
    main()
