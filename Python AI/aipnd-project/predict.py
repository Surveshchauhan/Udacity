import argparse
import os
import sys
import json
import ImageClassifier as ic

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict image using PyTorch')
    parser.add_argument('--gpu', action="store_true", dest="device", default=False)
    parser.add_argument('--image', action="store", dest="image")
    parser.add_argument('--checkpoint', action="store", dest="checkpoint", required=True)
    parser.add_argument('--top_k', action="store", dest="top_k", default=5, type=int)
    parser.add_argument('--category_names', action="store", dest="category_names")
  
    args = parser.parse_args()
    device = "cuda" if args.device == True else "cpu"

    if not os.path.exists(args.image):
        print('Unable to find image path to predict')
        sys.exit(-1)

    if not os.path.exists(args.checkpoint):
        print('Unable to find location checkpoint')
        sys.exit(-1)

    if args.top_k < 1:
        print('top_k should be greater than 0')
        sys.exit(-1)

    if not os.path.exists(args.category_names):
        print('Unable to find cateogry json file')
        sys.exit(-1)
 
    print()
    print('Loading JSON file...')
    
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)

    print("Loading saved checkpoint: " + args.checkpoint)
    model = ic.ImageClassifier()
    model.load(args.checkpoint)

    print()
    print('Inferring image:')
    result = model.predict(args.image, args.top_k)
    print()

    tags = result['tags']

    flowers = [cat_to_name[y] for _,y in tags]
    predictions = [x for x,_ in tags]

    print('-' * 50)
    print('{0:<15} {1}'.format('Probability','Image Name'))
    print('-' * 50)
    for x in zip(flowers, predictions):
        print('{0:<15.6f} {1}'.format(x[1], x[0]))
    print('-' * 50)
    print()
