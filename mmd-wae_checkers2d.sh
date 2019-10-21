
mkdir -p out

dt=$(date '+%d%m%Y%H%M%S');
echo $dt

rec_lambda=1.0
train_size=10000
exp=checkers
sinkhorn_iters=10
epoch_num=20

mmd_or_sinkhorn = 'mmd'
mmd_linear = False


for bs in 100 200 400 500 1000
do
for lr in 0.0001 0.0003
do
wae_lambda=0
ot_lambda=100
zdim=2
sinkhorn_iters=10
sinkhorn_epsilon=0.01
name="exp=${exp}_wae_bstuning_train_size=${train_size}_zdim=${zdim}_wae_lambda=${wae_lambda}_bs=${bs}_lr=${lr}_${dt}"
python run.py \
    --sinkhorn_sparse=False --sinkhorn_sparsifier=None --nat_sparse_indices_num=1000 \
    --pz=uniform --zdim=${zdim} \
    --enc_noise=deterministic --name="${name}" --nat_size=${train_size} \
    --nat_resampling=${nat_resampling} --tag="${exp},checkers_2_grid,wae,bstuning2" --exp=${exp} \
    --rec_lambda=${rec_lambda} --train_size=${train_size} --wae_lambda=${wae_lambda} --epoch_num=${epoch_num} \
    --batch_size=${bs} --lr=${lr} --frequency_of_latent_change=1 \
    --ot_lambda=0.0 --sinkhorn_epsilon=${sinkhorn_epsilon} --sinkhorn_iters=${sinkhorn_iters} \
    --e_pretrain=False --work_dir=out/wae_${name}  > out/wae_${name}.cout 2> out/wae_${name}.cerr
done
done
