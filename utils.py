import argparse
import torch
from ruamel.yaml import YAML

def get_args():
    yaml_path = 'configs.yaml'
    parser = argparse.ArgumentParser(description='Here is Meta Learning for Graph data.')

    parser.add_argument('--use_cuda', action='store_true', default=True)
    parser.add_argument('--data_path', type=str, default='./graphdata')
    parser.add_argument('--save_dir', type=str, default='./graphdata/save')

    parser.add_argument('--name', type=str, default='GFS')
    parser.add_argument('--exp_name', type=str, default='F')
    parser.add_argument('--model_name', type=str, default='GFS')
    parser.add_argument('--random_seed', type=int, default=231)
    parser.add_argument('--num_workers', type=int, default=0)

    parser.add_argument('--dataset', type=str, default='Cora')
    parser.add_argument('--num_samples', type=int, default=2708)
    parser.add_argument('--input_dim', type=str, default=1433)
    parser.add_argument('--num_classes', type=int, default=7)
    parser.add_argument('--num_negatives', type=int, default=27080, help='for queue')

    parser.add_argument('--max_epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--base_model', type=str, default='GCN')
    parser.add_argument('--activation', type=str, default="relu")
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--out_dim', type=int, default=128)
    parser.add_argument('--tau', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--best_pretrain', type=str, default='no.ckpt')

    parser.add_argument('--head_max_epochs', type=int, default=1)
    parser.add_argument('--head_hidden_dim', type=int, default=128)
    parser.add_argument('--head_lr', type=float, default=1e-2)
    parser.add_argument('--head_weight_decay', type=float, default=0)
    parser.add_argument('--best_head', type=str, default='no_head.ckpt')
    parser.add_argument('--classifier', type=str, default='LR')

    parser.add_argument('--class_split_ratio', type=list, default=[3, 2, 2])
    parser.add_argument('--n_way', type=int, default=2)
    parser.add_argument('--k_shot', type=int, default=5)
    parser.add_argument('--q_query', type=int, default=20)
    parser.add_argument('--task_num', type=int, default=50, help='Number of tasks used for test/validation.')
    parser.add_argument('--train_task_num', type=int, default=20, help='Number of tasks for calculating fs_loss')
    parser.add_argument('--temperature2', type=float, default=1)
    parser.add_argument('--fs_rate', type=float, default=1, help='Ratio of fs_loss in the final loss function.')

    parser.add_argument('--label_mask', type=int, default=0)
    parser.add_argument('--khop_mask', type=int, default=0)
    parser.add_argument('--self_mask', type=bool, default=False, help='Mask the node itself search process.')
    parser.add_argument('--k_rate', type=int, default=1)
    parser.add_argument('--mmt', type=float, default=0.9)
    parser.add_argument('--em_scd', type=int ,default=0)

    parser.add_argument('--f1', type=float, default=0.3, help='Augmentation ratio for the first feature.')
    parser.add_argument('--f2', type=float, default=0.4, help='Augmentation ratio for the second feature.')
    parser.add_argument('--f3', type=float, default=0.4, help='Augmentation ratio for the third feature.')
    parser.add_argument('--e1', type=float, default=0.2, help='Augmentation ratio for the first edge.')
    parser.add_argument('--e2', type=float, default=0.4, help='Augmentation ratio for the second edge.')
    parser.add_argument('--e3', type=float, default=0.4, help='Augmentation ratio for the third edge.')

    parser.add_argument('--compare_mode', type=str, default='m1')
    parser.add_argument('--model_mode', type=str, default='fs3')

    args = parser.parse_args()
    with open(yaml_path) as args_file:

        args_key = "-".join([args.dataset, args.model_name])
        print(args_key)
        try:
            parser.set_defaults(**dict(YAML().load(args_file)[args_key].items()))
        except KeyError:
            raise AssertionError('KeyError: there is no {} in yamls'.format(args_key), "red")

    args = parser.parse_args()

    return args


def precision_at_k(output, target, top_k=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k."""
    with torch.no_grad():
        maxk = max(top_k)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in top_k:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def map_class(classes: torch.Tensor, q_query: int) -> torch.Tensor:
    # map the true label to task label
    # classes: n_way
    # output: (q_query x n_way) x 1
    exp_classes = classes.unsqueeze(1).expand(classes.size(0), q_query).reshape(-1, 1).squeeze()
    map = {x.item(): i for i, x in enumerate(classes)}
    remap_classes = torch.LongTensor([map[x.item()] for x in exp_classes])
    return remap_classes

def normalize_0to1(tensor: torch.Tensor):
    min_values, _ = torch.min(tensor, dim=1, keepdim=True)
    max_values, _ = torch.max(tensor, dim=1, keepdim=True)
    normalized_tensor = (tensor - min_values) / (max_values - min_values)
    return normalized_tensor