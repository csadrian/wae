
mkdir -p out

dt=$(date '+%d%m%Y%H%M%S');
echo $dt

rec_lambda=1.0
train_size=100000
exp=celebA
nat_resampling=None
wae_lambda=0.0
epoch_num=100
pz=sphere
nat_size=-1
for sinkhorn_iters in 10
do
for sinkhorn_epsilon in 0.1
do
for ot_lambda in 0.1
do
for zdim in 64
do
# --train_size ${train_size}
#name="celeba_res=${nat_resampling}_exp=${exp}_logdomain_train_size=${train_size}_rec_lambda=${rec_lambda}_zdim=${zdim}__sinkhorn_iters=${sinkhorn_iters}_sinkhorn_epsilon=${sinkhorn_epsilon}_ot_lambda=${ot_lambda}_${dt}"
name="celeba_ours_random_eps=0.1_ot_lambda=0.1"
python run.py --pz=${pz} --train_size=${train_size} --sinkhorn_sparse=True --sinkhorn_sparsifier=random --nat_sparse_indices_num=40000  --enc_noise=deterministic --name="${name}" --nat_size=${nat_size} --nat_resampling=${nat_resampling} --tag="spherical,ordered-blocks,celeba" --exp=${exp} --rec_lambda=${rec_lambda} --wae_lambda=${wae_lambda} --epoch_num=${epoch_num} --ot_lambda=${ot_lambda} --sinkhorn_epsilon=${sinkhorn_epsilon} --sinkhorn_iters=${sinkhorn_iters} --zdim=${zdim} --e_pretrain=False --work_dir=out/sinkhorn_${name}  > out/sinkhorn_${name}.cout 2> out/sinkhorn_${name}.cerr
#python run.py --mode=generate --pz=${pz} --sinkhorn_sparse=False --sinkhorn_sparsifier=None  --nat_sparse_indices_num=1000  --enc_noise=deterministic --name="${name}" --nat_size=1000 --nat_resampling=${nat_resampling} --tag="spherical,ordered-blocks,mnist" --exp=${exp} --rec_lambda=${rec_lambda} --wae_lambda=${wae_lambda} --epoch_num=${epoch_num} --ot_lambda=${ot_lambda} --sinkhorn_epsilon=${sinkhorn_epsilon} --sinkhorn_iters=${sinkhorn_iters} --zdim=${zdim} --e_pretrain=False --work_dir=out/c > out/sinkhorn_${name}.cout 2> out/sinkhorn_${name}.cerr

done
done
done
done