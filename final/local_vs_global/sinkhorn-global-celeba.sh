mkdir -p out

dt=$(date '+%d%m%Y%H%M%S');
echo $dt

iteration_target=200000

i=0

for e_pretrain in False True
do
for nat_size in 5000
do
for bs in 100
do
for lr in 0.001
do
for wae_lambda in 1.0
do
for grad_clip in None 0.1
do

exp=celebA
zdim=64
pz=sphere

train_size=5000
recalculate_size=5000
#nat_size=${train_size}
epoch_num=$((iteration_target * bs / train_size))

z_test=sinkhorn
z_test_scope=global

rec_lambda=1.0
sinkhorn_iters=10
sinkhorn_epsilon=0.01

nat_resampling=None

name="sinkhorn-global-celeba_res=${nat_resampling}_exp=${exp}_batch_size=${bs}_lr=${lr}_train_size=${train_size}_rec_lambda=${rec_lambda}_sinkhorn_epsilon=${sinkhorn_epsilon}_wae_lambda=${wae_lambda}_${i}_${dt}"
CUDA_VISIBLE_DEVICES=${i} python run.py --pz=${pz} --train_size=${train_size} \
    --z_test=${z_test} --z_test_scope=${z_test_scope} \
    --sinkhorn_sparse=False --enc_noise=deterministic --name="${name}" \
    --recalculate_size=${recalculate_size} --nat_size=${nat_size} --nat_resampling=${nat_resampling} --tag="sinkhorn,local_vs_global,global,final0" \
    --exp=${exp} --rec_lambda=${rec_lambda} --wae_lambda=${wae_lambda} --epoch_num=${epoch_num} \
    --sinkhorn_epsilon=${sinkhorn_epsilon} --sinkhorn_iters=${sinkhorn_iters} \
    --zdim=${zdim} --e_pretrain=${e_pretrain} --batch_size=${bs} --lr=${lr} --work_dir=out/sinkhorn_${name} \
    --shuffle=True --grad_clip=${grad_clip} \
        > out/${name}.cout 2> out/${name}.cerr &

((i=i+1))
done
done
done
done
done
done