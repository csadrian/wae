mkdir -p out

dt=$(date '+%d%m%Y%H%M%S');
echo $dt

iteration_target=20000
for nat_size in 1000 5000 10000 50000
do
for bs in 50 100
do
for lr in 0.001
do
for wae_lambda in 0.0001 0.001 0.01 0.1
do

exp=mnist_ord
zdim=8
pz=normal

train_size=50000
recalculate_size=${bs}
#nat_size=${train_size}
epoch_num=$((iteration_target * bs / train_size))

z_test=sinkhorn
z_test_scope=global

rec_lambda=1.0
sinkhorn_iters=10
sinkhorn_epsilon=0.01

nat_resampling=None
# loss_mmd (a monitored but not optimized value) is calculated with this many prior samples:

name="sinkhorn-nu-global-mnist_res=${nat_resampling}_exp=${exp}_batch_size=${bs}_lr=${lr}_train_size=${train_size}_rec_lambda=${rec_lambda}_sinkhorn_epsilon=${sinkhorn_epsilon}_wae_lambda=${wae_lambda}_${dt}"
CUDA_VISIBLE_DEVICES=2 python run.py --pz=${pz} --train_size=${train_size} \
    --z_test=${z_test} --z_test_scope=${z_test_scope} \
    --sinkhorn_sparse=False --enc_noise=deterministic --name="${name}" \
    --recalculate_size=${recalculate_size} --nat_size=${nat_size} --nat_resampling=${nat_resampling} --tag="mnist-nonshuffled,global-sinkhorn" \
    --exp=${exp} --rec_lambda=${rec_lambda} --wae_lambda=${wae_lambda} --epoch_num=${epoch_num} \
    --sinkhorn_epsilon=${sinkhorn_epsilon} --sinkhorn_iters=${sinkhorn_iters} \
    --zdim=${zdim} --e_pretrain=False --batch_size=${bs} --lr=${lr} --work_dir=out/sinkhorn_${name} \
    --shuffle=False \
        > out/${name}.cout 2> out/${name}.cerr
done
done
done
done