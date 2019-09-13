
mkdir -p out

dt=$(date '+%d%m%Y%H%M%S');
echo $dt

rec_lambda=1.0
train_size=2000
exp=syn_2_constant_uniform
sinkhorn_iters=10
epoch_num=10

for nat_resampling in None batch
do
for sinkhorn_epsilon in 0.01 0.1 1 10
do
for ot_lambda in 0 0.0001 0.001 0.01 0.1 1 10
do
for zdim in 2
do
name="exp=${exp}_global_res=${nat_resampling}_train_size=${train_size}_rec_lambda=${rec_lambda}_zdim=${zdim}_sinkhorn_epsilon=${sinkhorn_epsilon}_ot_lambda=${ot_lambda}_${dt}"
python run.py \
    --sinkhorn_sparse=False --sinkhorn_sparsifier=None --nat_sparse_indices_num=1000 \
    --pz=uniform --zdim=${zdim} \
    --enc_noise=deterministic --name="${name}" --nat_size=${train_size} \
    --nat_resampling=${nat_resampling} --tag="${exp},syn_2_grid,global_dense" --exp=${exp} \
    --rec_lambda=${rec_lambda} --train_size=${train_size} --wae_lambda=0.0 --epoch_num=${epoch_num} \
    --ot_lambda=${ot_lambda} --sinkhorn_epsilon=${sinkhorn_epsilon} --sinkhorn_iters=${sinkhorn_iters} \
    --e_pretrain=False --work_dir=out/sinkhorn_${name}  > out/sinkhorn_${name}.cout 2> out/sinkhorn_${name}.cerr
done
done
done
done
