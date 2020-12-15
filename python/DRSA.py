from BASE_MODEL import BASE_RNN
import sys
import argparse

def restricted_float(x):
    try:
        x = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError("%r not a floating-point literal" % (x,))

    if x < 0.0 or x > 1.0:
        raise argparse.ArgumentTypeError("%r not in range [0.0, 1.0]"%(x,))
    return x

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', help="Dataset", default = "support")
    parser.add_argument('learning_rate', type =restricted_float, help = "Input learning rate", default = 0.0001)
    parser.add_argument('FEATURE_SIZE', help = "dataset input fields count", type = int)
    parser.add_argument('MAX_DEN', help = "max input data dimension", default = 580000, type = int)
    parser.add_argument('EMB_DIM',help='EMB Dimension', default=32, type = int)
    parser.add_argument('BATCH_SIZE', default = 128, help="Batch size", type = int)
    parser.add_argument('MAX_SEQ_LEN', default = 330, help="Maximum sequence length", type = int)
    parser.add_argument('STATE_SIZE', default = 128, help="State size", type = int)
    parser.add_argument('GRAND_CLIP', default = 5.0, help="Grand clip")
    parser.add_argument('--L2_NORM', default = 0.001, help="L2 Normalizing coefficient (?)", type = restricted_float)
    parser.add_argument('--ADD_TIME', action="store_true", help="If arg present, add time")
    parser.add_argument('--ALPHA', default=1.2, help='Coefficient for cross entropy')
    parser.add_argument('--BETA', default=0.2, help='Coefficient for ANLP')
    parser.add_argument('--TRAING_STEPS', type=int, default=10000000, help="Number of training steps. Default: 10000000")
    return parser.parse_args()

print("Starting DRSA script")
if __name__ == '__main__':
    args = parse_args()
    print("Arguments:",args)

if len(sys.argv) < 2:
    print("Please input learning rate. ex. 0.0001")
    sys.exit(0)

LR = float(args.learning_rate)
print("Learning rate is:", LR)
LR_ANLP = LR
RUNNING_MODEL = BASE_RNN(EMB_DIM=args.EMB_DIM,
                         FEATURE_SIZE=args.FEATURE_SIZE,
                         BATCH_SIZE=args.BATCH_SIZE,
                         MAX_DEN=args.MAX_DEN,
                         MAX_SEQ_LEN=args.MAX_SEQ_LEN,
                         TRAING_STEPS=args.TRAING_STEPS,
                         STATE_SIZE=args.STATE_SIZE,
                         LR=LR,
                         GRAD_CLIP=args.GRAD_CLIP,
                         L2_NORM=args.L2_NORM,
                         INPUT_FILE=args.input_file,
                         ALPHA=args.ALPHA,
                         BETA=args.BETA,
                         ADD_TIME_FEATURE=args.ADD_TIME,
                         FIND_PARAMETER=False,
                         ANLP_LR=LR,
                         DNN_MODEL=False,
                         DISCOUNT=1,
                         ONLY_TRAIN_ANLP=False,
                         LOG_PREFIX="drsa")
print("Start of CREATE_GRAPH")
RUNNING_MODEL.create_graph()
print("END OF CREATE_GRAPH")
print("Start of RUN_MODEL")
RUNNING_MODEL.run_model()
print("END OF RUN_MODEL")
print("DRSA dataset:", args.input_file)
print("END of DRSA")
