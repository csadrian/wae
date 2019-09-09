
mkdir -p out

dt=$(date '+%d%m%Y%H%M%S');
echo $dt

rec_lambda=0.0
train_size=100
exp=mnist
nat_resampling=None
for sinkhorn_iters in 10
do
for sinkhorn_epsilon in 0.1
do
for ot_lambda in 1.0
do
for zdim in 8
do
name="mnist_res=${nat_resampling}_exp=${exp}_logdomain_train_size=${train_size}_rec_lambda=${rec_lambda}_zdim=${zdim}__sinkhorn_iters=${sinkhorn_iters}_sinkhorn_epsilon=${sinkhorn_epsilon}_ot_lambda=${ot_lambda}_${dt}"
python run.py --sinkhorn_sparse=True --sinkhorn_sparsifier=full  --nat_sparse_indices_num=10000  --enc_noise=deterministic --name="${name}" --nat_size=100 --nat_resampling=${nat_resampling} --tag="spherical,ordered-blocks,mnist" --exp=${exp} --rec_lambda=${rec_lambda} --train_size ${train_size} --wae_lambda=0.0 --epoch_num=300 --ot_lambda=${ot_lambda} --sinkhorn_epsilon=${sinkhorn_epsilon} --sinkhorn_iters=${sinkhorn_iters} --zdim=${zdim} --e_pretrain=False --work_dir=out/sinkhorn_${name}  > out/sinkhorn_${name}.cout 2> out/sinkhorn_${name}.cerr
done
done
done
done