import argparse
import torch

import model_service as ms


def get_args():
    parser = argparse.ArgumentParser(description="Model prediction settings")
    
    parser.add_argument('image_path', type=str, help="Path to image to be classified (required)")
    parser.add_argument('checkpoint', type=str, help="Path to model training checkpoint to be used for the classification process (required)")
    
    parser.add_argument('--top_k', default=1, type=int, help='Model architecture')
    parser.add_argument('--category_names', default='', type=str, help='Path to directory where checkpoint will be saved')
    parser.add_argument('--gpu', default=False, action='store_true', help='Use GPU for training')

    args = parser.parse_args()

    return args


def main():
    args = get_args()
    
    checkpoint = ms.load_checkpoint(args.checkpoint)
    
    env = 'cpu'
    
    if args.gpu:
        if torch.cuda.is_available():
            env = 'cuda'
        else:
            print('Your environment has no GPU support...\nAll calculations will run into CPU, which may slow down the prediction.')
            
    probs, labels = ms.predict(args.image_path, checkpoint, args.top_k, args.category_names, env)
    
    for prob, label in zip(probs, labels):
        print("{}% of chance to be a {}".format(prob, label))


if __name__ == '__main__':
    main()
