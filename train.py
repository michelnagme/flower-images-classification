import argparse
import torch

import model_service as ms


def get_args():
    parser = argparse.ArgumentParser(description="Model training settings")
    
    parser.add_argument('data_dir', type=str, help="Path to directory where the data is located (required)")
    
    parser.add_argument('--arch', default='vgg16', type=str, help='Model architecture')
    parser.add_argument('--save_dir', default='', type=str, help='Path to directory where checkpoint will be saved')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='Gradient descent learning rate')
    parser.add_argument('--hidden_units', default=1024, type=int, help='Number of hidden units for the classifier')
    parser.add_argument('--epochs', default=3, type=int, help='Number of epochs for training')
    parser.add_argument('--gpu', default=False, action='store_true', help='Use GPU for training')

    args = parser.parse_args()

    return args


def main():
    args = get_args()
    
    image_datasets = ms.get_image_datasets(args.data_dir)
    dataloaders = ms.get_dataloaders(image_datasets)
    
    env = 'cpu'
    
    if args.gpu:
        if torch.cuda.is_available():
            env = 'cuda'
        else:
            print('Your environment has no GPU support...\nAll calculations will run into CPU, which may slow down the training.')
    
    model = ms.model_train(args.arch, args.epochs, env, args.hidden_units, args.learning_rate, dataloaders)
    
    if model:
        checkpoint = {
            'arch': args.arch,
            'hidden_units': args.hidden_units,
            'class_to_idx': image_datasets['train_data'].class_to_idx,
            'state_dict': model.state_dict()
        }

        torch.save(checkpoint, '{}/checkpoint_{}.pth'.format('.' if not args.save_dir else args.save_dir, args.arch))


if __name__ == '__main__':
    main()
