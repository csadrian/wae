name="celeba_z_test=sinkhorn_z_test_scope=local_bs=200_train=10000_proj=1_2_lr=0.00001_wae_lambda=0.1" ; CUDA_VISIBLE_DEVICES=0 python run.py --z_test=sinkhorn --z_test_scope=local --sinkhorn_sparse=False
--sinkhorn_sparsifier=None --batch_size=200 --enc_noise=deterministic --name=$name --nat_size=10000 --nat_resampling=None --tag="sinkhorn" --exp=celebA --pz=normal --rec_lambda=1.0
--train_size=10000 --wae_lambda=0.1 --sinkhorn_epsilon=0.01 --recalculate_size=10000 --sinkhorn_iters=10 --e_pretrain=False --work_dir=out/sinkhorn_$name
