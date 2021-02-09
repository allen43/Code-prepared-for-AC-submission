if __name__ == '__main__':
    from time import time
    t00 = time()
    import numpy as np
    import torch.nn as nn
    #x_fn为数据，3000*1000，y_fn为label数据，3000*1

    X = np.loadtxt('E:\zhoubo/data/data_nc_finetune.txt')
    y = np.loadtxt('E:\zhoubo/data/label_nc_finetune.txt')
    print(X.shape, y.shape)


    from resnet import ResNet
    import os
    import torch
    # CNN parameters
    layers = 6
    hidden_size = 100
    block_size = 2
    hidden_sizes = [hidden_size] * layers
    num_blocks = [block_size] * layers
    #input_dim为单个细菌数据的维度。我的为861
    input_dim = 1000
    in_channels = 64
    #类别数目，我的是15
    n_classes = 30
    os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(0)
    cuda = torch.cuda.is_available()

    # Load trained weights for demo
    cnn = ResNet(hidden_sizes, num_blocks, input_dim=input_dim,
                    in_channels=in_channels, n_classes=n_classes)
    if cuda: cnn.cuda()
    cnn.load_state_dict(torch.load(
        './pretrained_model.ckpt', map_location=lambda storage, loc: storage))
    num_nerual = cnn.linear.in_features
    cnn.linear = nn.Linear(num_nerual,8)

    from datasets import spectral_dataloader
    from training import run_epoch
    from torch import optim
    for i in range(5):
        p_val = 0.1
        n_val = int(800 * p_val)
        idx_tr = list(range(800))
        np.random.shuffle(idx_tr)
        idx_val = idx_tr[:n_val]   #验证集的index
        idx_tr = idx_tr[n_val:]     #训练集的Index


        # Fine-tune CNN
        epochs = 30 #Change this number to ~30 for full training
        batch_size = 10
        t0 = time()  #计时开始
        # Set up Adam optimizer
        optimizer = optim.Adam(cnn.parameters(), lr=1e-3, betas=(0.5, 0.999))
        # Set up dataloaders
        dl_tr = spectral_dataloader(X, y, idxs=idx_tr,
            batch_size=batch_size, shuffle=True)
        dl_val = spectral_dataloader(X, y, idxs=idx_val,
            batch_size=batch_size, shuffle=False)
        # Fine-tune CNN for first fold
        best_val = 0
        no_improvement = 0
        max_no_improvement = 5
        print('Starting fine-tuning!')
        for epoch in range(epochs):
            print(' Epoch {}: {:0.2f}s'.format(epoch+1, time()-t0))
            # Train
            acc_tr, loss_tr = run_epoch(epoch, cnn, dl_tr, cuda,
                training=True, optimizer=optimizer)
            print('  Train acc: {:0.2f}'.format(acc_tr))
            # Val
            acc_val, loss_val = run_epoch(epoch, cnn, dl_val, cuda,
                training=False, optimizer=optimizer)
            print('  Val acc  : {:0.2f}'.format(acc_val))
            # Check performance for early stopping
            if acc_val > best_val or epoch == 0:
                best_val = acc_val
                torch.save(cnn, './model/finetuned_model'+str(i)+'.pth')
                no_improvement = 0
            else:
                no_improvement += 1
            if no_improvement >= max_no_improvement:
                print('第'+str(i)+'循环最高acc_val',best_val)
                print('Finished after {} epochs!'.format(epoch+1))
                break

    print('\n This demo was completed in: {:0.2f}s'.format(time()-t00))
