import argparse
import os
import signal
import subprocess
import time

import torch
import torch.optim as optim
import torch.nn as nn
import copy
torch.set_num_threads(1)

import syft
import syft as sy
from syft.serde.compression import NO_COMPRESSION
from syft.grid.clients.data_centric_fl_client import DataCentricFLClient

sy.serde.compression.default_compress_scheme = NO_COMPRESSION

from procedure import train, test
from data import get_data_loaders, get_number_classes, get_federated_data_loaders
from models import get_model, load_state_dict
from preprocess import build_prepocessing
 
NUM_CLIENTS =16
BATCH_SIZE = 64
NUM_GROUPS = int(NUM_CLIENTS / 2)

def run(args):
    if args.train:
        print(f"Training over {args.global_epochs} global epochs")
        print(f"Training over {args.local_epochs} local epochs")
    elif args.test:
        print("Running a full evaluation")
    else:
        print("Running inference speed test")
    print("model:\t\t", args.model)
    print("dataset:\t", args.dataset)
    print("batch_size:\t", args.batch_size)
    print("Num of clients:\t", NUM_CLIENTS)
    print("Num of groups:\t", NUM_GROUPS)

    hook = sy.TorchHook(torch)

    workers = [None] * NUM_CLIENTS * args.global_epochs
    encryption_kwargs = [None] * NUM_CLIENTS * args.global_epochs
    kwargs = [None] * NUM_CLIENTS * args.global_epochs
    
    totModel = None
    modeltot = get_model(args.model, args.dataset, out_features=get_number_classes(args.dataset))


    #################### Here I create the virtual workers for the global model #######################
    jack = sy.VirtualWorker(hook, id="jack")
    john = sy.VirtualWorker(hook, id="john")
    crypto_provider1 = sy.VirtualWorker(hook, id="crypto_provider")
    workers_a = [jack, john]
    sy.local_worker.clients = workers_a
    encryption_kwargs1 = dict(
        workers=workers_a, crypto_provider=crypto_provider1, protocol=args.protocol
    )
    kwargs1 = dict(
        requires_grad=args.requires_grad,
        precision_fractional=args.precision_fractional,
        dtype=args.dtype,
        **encryption_kwargs1,
    )
    global_model = copy.deepcopy(modeltot)
    if not args.public:
        global_model.encrypt(**kwargs1)
        if args.fp_only:  # Just keep the (Autograd+) Fixed Precision feature
            global_model.get()

    #################### end of generating global model #######################

    
    # public_train_loader, public_test_loader = get_data_loaders(args, kwargs, private=False)
    model = [None] * NUM_CLIENTS * args.global_epochs
    for gepoch in range(args.global_epochs):
        print("running training global epoch" + str(gepoch))
        global_model.decrypt()
        new_global = copy.deepcopy(global_model)
        if not args.public:
            global_model.encrypt(**kwargs1)
            if args.fp_only:  # Just keep the (Autograd+) Fixed Precision feature
                global_model.get()

        for globalClients in range(NUM_GROUPS):
            print("running training on group" + str(globalClients))
            local_model = copy.deepcopy(new_global)
            for localClients in range(2):
                
                client = globalClients * 2 + localClients
                print("running training for the data of client" + str(client))
                bob = sy.VirtualWorker(hook, id="bob"+str(gepoch*NUM_CLIENTS + client))
                alice = sy.VirtualWorker(hook, id="alice"+str(gepoch*NUM_CLIENTS + client))
                crypto_provider = sy.VirtualWorker(hook, id="crypto_provider"+ str(gepoch*NUM_CLIENTS + client))
                workers[gepoch*NUM_CLIENTS + client] = [alice, bob]
                sy.local_worker.clients = workers[gepoch*NUM_CLIENTS + client]

                encryption_kwargs[gepoch*NUM_CLIENTS + client] = dict(
                    workers=workers[gepoch*NUM_CLIENTS + client], crypto_provider=crypto_provider, protocol=args.protocol
                )
                kwargs[gepoch*NUM_CLIENTS + client] = dict(
                    requires_grad=args.requires_grad,
                    precision_fractional=args.precision_fractional,
                    dtype=args.dtype,
                    **encryption_kwargs[gepoch*NUM_CLIENTS + client],
                )
                if args.preprocess:
                    build_prepocessing(args.model, args.dataset, args.batch_size, workers[gepoch*NUM_CLIENTS + client], args)

                federated_train_loaders, new_test_loaders = get_federated_data_loaders(args, kwargs[gepoch*NUM_CLIENTS + client], num_clients=NUM_CLIENTS, private=True)
                # new_global.decrypt()
                model[gepoch*NUM_CLIENTS + client] = copy.deepcopy(new_global)
                # if not args.public:
                #     new_global.encrypt(**kwargs1)
                #     if args.fp_only:  # Just keep the (Autograd+) Fixed Precision feature
                #         new_global.get()
                if args.test and not args.train:
                    load_state_dict(model[gepoch*NUM_CLIENTS + client], args.model, args.dataset)

                model[gepoch*NUM_CLIENTS + client].eval()
                if torch.cuda.is_available():
                    sy.cuda_force = True

                print(model[gepoch*NUM_CLIENTS + client].fc1.weight.data)
                        
                if not args.public:
                    model[gepoch*NUM_CLIENTS + client].encrypt(**kwargs[gepoch*NUM_CLIENTS + client])
                    if args.fp_only:  # Just keep the (Autograd+) Fixed Precision feature
                        model[gepoch*NUM_CLIENTS + client].get()

                if args.train:
                    for epoch in range(args.local_epochs):
                        optimizer = optim.SGD(model[gepoch*NUM_CLIENTS + client].parameters(), lr=args.lr, momentum=args.momentum)

                        if not args.public:
                            optimizer = optimizer.fix_precision(
                                precision_fractional=args.precision_fractional, dtype=args.dtype
                            )
                        print("start train")
                        train_time = train(args, model[gepoch*NUM_CLIENTS + client], federated_train_loaders[client], optimizer, epoch)
                        print("end train")
                    
                if localClients == 0:

                    model[gepoch*NUM_CLIENTS + client].decrypt()
                    print("local_model 0")
                    print(model[gepoch*NUM_CLIENTS + client].fc1.weight.data)
                    local_model = copy.deepcopy(model[gepoch*NUM_CLIENTS + client])
                    if not args.public:
                        # model[client].encrypt(**kwargs[client])
                        local_model.encrypt(**kwargs1)
                        if args.fp_only:  # Just keep the (Autograd+) Fixed Precision feature
                            # model[client].get()
                            local_model.get()

                    # print("alice has: " + str(alice._objects))
                    # model[client].fc1.weight.data=model[client].fc1.weight.data.send(john)

                else:

                    model[gepoch*NUM_CLIENTS + client].decrypt()
                    print("local_model 1")
                    print(model[gepoch*NUM_CLIENTS + client].fc1.weight.data)
                    new_model = copy.deepcopy(model[gepoch*NUM_CLIENTS + client])
                    if not args.public:
                        # model[client].encrypt(**kwargs[client])
                        new_model.encrypt(**kwargs1)
                        if args.fp_only:  # Just keep the (Autograd+) Fixed Precision feature
                            # model[client].get()
                            new_model.get()

                    # print("alice has: " + str(alice._objects))
                    if args.model == 'network1':
                        local_model.fc1.weight.data = (local_model.fc1.weight.data + new_model.fc1.weight.data)
                        local_model.fc2.weight.data = (local_model.fc2.weight.data + new_model.fc2.weight.data)
                        local_model.fc3.weight.data = (local_model.fc3.weight.data + new_model.fc3.weight.data)
                        local_model.fc1.bias.data = (local_model.fc1.bias.data + new_model.fc1.bias.data)
                        local_model.fc2.bias.data = (local_model.fc2.bias.data + new_model.fc2.bias.data)
                        local_model.fc3.bias.data = (local_model.fc3.bias.data + new_model.fc3.bias.data)
                    elif args.model == 'network2':
                        local_model.conv1.weight.data = local_model.conv1.weight.data + new_model.conv1.weight.data
                        local_model.conv2.weight.data = local_model.conv2.weight.data + new_model.conv2.weight.data
                        local_model.fc1.weight.data = local_model.fc1.weight.data +new_model.fc1.weight.data
                        local_model.fc2.weight.data = local_model.fc2.weight.data + new_model.fc2.weight.data
                    elif args.model == 'lenet':
                        local_model.conv1.weight.data = local_model.conv1.weight.data + new_model.conv1.weight.data
                        local_model.conv2.weight.data = local_model.conv2.weight.data + new_model.conv2.weight.data
                        local_model.fc1.weight.data = local_model.fc1.weight.data +new_model.fc1.weight.data
                        local_model.fc2.weight.data = local_model.fc2.weight.data + new_model.fc2.weight.data

            if globalClients == 0:
                if args.model == 'network1':
                    global_model.fc1.weight.data = local_model.fc1.weight.data
                    global_model.fc2.weight.data = local_model.fc2.weight.data
                    global_model.fc3.weight.data = local_model.fc3.weight.data
                    global_model.fc1.bias.data = local_model.fc1.bias.data
                    global_model.fc2.bias.data = local_model.fc2.bias.data
                    global_model.fc3.bias.data = local_model.fc3.bias.data
                elif args.model == 'network2':
                    global_model.conv1.weight.data = local_model.conv1.weight.data
                    global_model.conv2.weight.data = local_model.conv2.weight.data
                    global_model.fc1.weight.data = local_model.fc1.weight.data
                    global_model.fc2.weight.data = local_model.fc2.weight.data
                elif args.model == 'lenet':
                    global_model.conv1.weight.data = local_model.conv1.weight.data
                    global_model.conv2.weight.data = local_model.conv2.weight.data
                    global_model.fc1.weight.data = local_model.fc1.weight.data
                    global_model.fc2.weight.data = local_model.fc2.weight.data


                # just for print can be removed
                
                global_model.decrypt()
                print("global model 0")
                print(global_model.fc1.weight.data)
                if not args.public:
                    global_model.encrypt(**kwargs1)
                    if args.fp_only:
                        global_model.get()

                # end of just for print can be removed
            else: 
                if args.model == 'network1':
                    global_model.fc1.weight.data = (global_model.fc1.weight.data + local_model.fc1.weight.data)
                    global_model.fc2.weight.data = (global_model.fc2.weight.data + local_model.fc2.weight.data)
                    global_model.fc3.weight.data = (global_model.fc3.weight.data + local_model.fc3.weight.data)
                    global_model.fc1.bias.data = (global_model.fc1.bias.data + local_model.fc1.bias.data)
                    global_model.fc2.bias.data = (global_model.fc2.bias.data + local_model.fc2.bias.data)
                    global_model.fc3.bias.data = (global_model.fc3.bias.data + local_model.fc3.bias.data)
                elif args.model == 'network2':
                    global_model.conv1.weight.data += local_model.conv1.weight.data
                    global_model.conv2.weight.data += local_model.conv2.weight.data
                    global_model.fc1.weight.data += local_model.fc1.weight.data
                    global_model.fc2.weight.data += local_model.fc2.weight.data
                elif args.model == 'lenet':
                    global_model.conv1.weight.data += local_model.conv1.weight.data
                    global_model.conv2.weight.data += local_model.conv2.weight.data
                    global_model.fc1.weight.data += local_model.fc1.weight.data
                    global_model.fc2.weight.data += local_model.fc2.weight.data

                # just for print can be removed
                
                global_model.decrypt()
                print("global model n")
                print(global_model.fc1.weight.data)
                if not args.public:
                    global_model.encrypt(**kwargs1)
                    if args.fp_only:
                        global_model.get()

                # end of just for print can be removed
            
            if globalClients ==(NUM_GROUPS-1):
                if args.model == 'network1':
                    global_model.fc1.weight.data = (global_model.fc1.weight.data )/ NUM_CLIENTS
                    global_model.fc2.weight.data = (global_model.fc2.weight.data )/ NUM_CLIENTS
                    global_model.fc3.weight.data = (global_model.fc3.weight.data )/NUM_CLIENTS
                    global_model.fc1.bias.data = (global_model.fc1.bias.data )/ NUM_CLIENTS
                    global_model.fc2.bias.data = (global_model.fc2.bias.data )/ NUM_CLIENTS
                    global_model.fc3.bias.data = (global_model.fc3.bias.data )/NUM_CLIENTS
                elif args.model == 'network2':
                    global_model.fc1.weight.data /= NUM_CLIENTS
                    global_model.fc2.weight.data /= NUM_CLIENTS
                    global_model.conv1.weight.data /= NUM_CLIENTS
                    global_model.conv2.weight.data /= NUM_CLIENTS
                elif args.model == 'lenet':
                    global_model.fc1.weight.data /= NUM_CLIENTS
                    global_model.fc2.weight.data /= NUM_CLIENTS
                    global_model.conv1.weight.data /= NUM_CLIENTS
                    global_model.conv2.weight.data /= NUM_CLIENTS
                
            # model[client].get(workers_a)
        if args.preprocess:
            missing_items = [len(v) for k, v in sy.preprocessed_material.items()]
            if sum(missing_items) > 0:
                print("MISSING preprocessed material")
                for key, value in sy.preprocessed_material.items():
                    print(f"'{key}':", value, ",")

        # if not args.public:
        #     decmodel0.encrypt(**kwargs)
        #     if args.fp_only:  # Just keep the (Autograd+) Fixed Precision feature
        #         decmodel0.get()
        print("evaluation")
        global_model.decrypt()
        print("global model final")
        print(global_model.fc1.weight.data)
        new_model.decrypt()
        if not args.public:
            global_model.encrypt(**kwargs1)
            new_model.encrypt(**kwargs1)
            if args.fp_only:  # Just keep the (Autograd+) Fixed Precision feature
                global_model.get()
                new_model.get()
        private_train_loader, private_test_loader = get_data_loaders(args, kwargs1, private=True)
        test_time, accuracy = test(args, global_model, private_test_loader)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        help="model to use for inference (network1, network2, lenet, alexnet, vgg16, resnet18)",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        help="dataset to use (mnist, cifar10, hymenoptera, tiny-imagenet)",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        help="size of the batch to use. Default 128.",
        default=64,
    )
    parser.add_argument(
        "--test_batch_size",
        type=int,
        help="size of the batch to use",
        default=None,
    )

    parser.add_argument(
        "--preprocess",
        help="[only for speed test] preprocess data or not",
        action="store_true",
    )

    parser.add_argument(
        "--fp_only",
        help="Don't secret share values, just convert them to fix precision",
        action="store_true",
    )

    parser.add_argument(
        "--public",
        help="[needs --train] Train without fix precision or secret sharing",
        action="store_true",
    )

    parser.add_argument(
        "--test",
        help="run testing on the complete test dataset",
        action="store_true",
    )

    parser.add_argument(
        "--train",
        help="run training for n epochs",
        action="store_true",
    )

    parser.add_argument(
        "--gepochs",
        type=int,
        help="[needs --train] number of global epochs to train on. Default 15.",
        default=2,
    )
    parser.add_argument(
        "--lepochs",
        type=int,
        help="[needs --train] number of local epochs to train on. Default 15.",
        default=8,
    )

    parser.add_argument(
        "--lr",
        type=float,
        help="[needs --train] learning rate of the SGD. Default 0.01.",
        default=0.01,
    )

    parser.add_argument(
        "--momentum",
        type=float,
        help="[needs --train] momentum of the SGD. Default 0.9.",
        default=0.9,
    )

    parser.add_argument(
        "--websockets",
        help="use PyGrid nodes instead of a virtual network. (nodes are launched automatically)",
        action="store_true",
    )

    parser.add_argument(
        "--verbose",
        help="show extra information and metrics",
        action="store_true",
    )

    parser.add_argument(
        "--log_interval",
        type=int,
        help="[needs --test or --train] log intermediate metrics every n batches. Default 10.",
        default=10,
    )

    parser.add_argument(
        "--comm_info",
        help="Print communication information",
        action="store_true",
    )

    parser.add_argument(
        "--pyarrow_info",
        help="print information about PyArrow usage and failure",
        action="store_true",
    )

    cmd_args = parser.parse_args()

    # Sanity checks

    if cmd_args.test or cmd_args.train:
        assert (
            not cmd_args.preprocess
        ), "Can't preprocess for a full epoch evaluation or training, remove --preprocess"

    if cmd_args.train:
        assert not cmd_args.test, "Can't set --test if you already have --train"

    if cmd_args.fp_only:
        assert not cmd_args.preprocess, "Can't have --preprocess in a fixed precision setting"
        assert not cmd_args.public, "Can't have simultaneously --fp_only and --public"

    if not cmd_args.train:
        assert not cmd_args.public, "--public is used only for training"

    if cmd_args.pyarrow_info:
        sy.pyarrow_info = True

    class Arguments:
        model = cmd_args.model.lower()
        dataset = cmd_args.dataset.lower()
        preprocess = cmd_args.preprocess
        websockets = cmd_args.websockets
        verbose = cmd_args.verbose

        train = cmd_args.train
        n_train_items = -1 if cmd_args.train else cmd_args.batch_size
        test = cmd_args.test or cmd_args.train
        n_test_items = -1 if cmd_args.test or cmd_args.train else cmd_args.batch_size

        batch_size = cmd_args.batch_size
        # Defaults to the train batch_size
        test_batch_size = cmd_args.test_batch_size or cmd_args.batch_size

        log_interval = cmd_args.log_interval
        comm_info = cmd_args.comm_info

        local_epochs = cmd_args.lepochs
        global_epochs = cmd_args.gepochs
        lr = cmd_args.lr
        momentum = cmd_args.momentum

        public = cmd_args.public
        fp_only = cmd_args.fp_only
        requires_grad = cmd_args.train
        dtype = "long"
        protocol = "fss"
        precision_fractional = 5 if cmd_args.train else 4

    args = Arguments()

    if args.websockets:
        print("Launching the websocket workers...")

        def kill_processes(worker_processes):
            for worker_process in worker_processes:
                pid = worker_process.pid
                try:
                    os.killpg(os.getpgid(worker_process.pid), signal.SIGTERM)
                    print(f"Process {pid} killed")
                except ProcessLookupError:
                    print(f"COULD NOT KILL PROCESS {pid}")

        worker_processes = [
            subprocess.Popen(
                f"./scripts/launch_{worker}.sh",
                stdout=subprocess.PIPE,
                shell=True,
                preexec_fn=os.setsid,
                executable="/bin/bash",
            )
            for worker in ["alice", "bob", "crypto_provider"]
        ]
        time.sleep(7)
        try:
            print("LAUNCHED", *[p.pid for p in worker_processes])
            run(args)
            kill_processes(worker_processes)
        except Exception as e:
            kill_processes(worker_processes)
            raise e

    else:
        # arg= [None] * NUM_CLIENTS
        # totalModel = Null
        # for client in range(NUM_CLIENTS):
            # arg[client] = Arguments()
        # for client in range(NUM_CLIENTS):
            # print("running training on client" + str(client))
        run(args)
            # print("done on client" + str(client))
            # totalModel += arg[client].model
