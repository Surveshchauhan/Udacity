import argparse
import os
import sys
import ImageClassifier as ic

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train image classifier')
    parser.add_argument('--gpu', action="store_true", dest="device", default=False)
    parser.add_argument('--learning_rate', action="store", dest="learning_rate", type=float, default=0.01)
    parser.add_argument('--save_dir', action="store", dest="save_dir", default=".")
    parser.add_argument('--data_dir', action="store", dest="data_dir", default="data_dir")
    parser.add_argument('--epochs', action="store", dest="epochs", type=int, default=5)
    parser.add_argument('--num_classes', action="store", type=int, dest="num_classes", default=10)

    args = parser.parse_args()
    device = "cuda" if args.device == True else "cpu"

    if not os.path.exists(args.data_dir):
        print('Unable to find training data folder')
        sys.exit(-1)

    if not os.path.exists(args.save_dir):
        print('Unable to find location to save trained model')
        sys.exit(-1)

    if args.epochs < 1 or args.epochs > 10000:
        print('Acceptable epochs is between 1 and 10000')
        sys.exit(-1)

    if args.num_classes < 1:
        print('num_classes should be greater than zero (0)')
        sys.exit(-1)

    if args.learning_rate < 0:
        print('learning_rate should be a positive number')
        sys.exit(-1)
 
    print()
    print("Creating model...")
    model = ic.ImageClassifier(learn_rate=args.learning_rate, num_output=args.num_classes, compute_device=device)
    
    print("Training model from " + args.data_dir)
    print()
    checkpoint_path = os.path.join(args.save_dir, "checkpoint.pth")
    model.load(checkpoint_path)

    model.train(args.data_dir, args.epochs, 20)

    print('Saving model to '+ checkpoint_path)
    model.save(checkpoint_path)
    
    print('Operation completed...')



 
