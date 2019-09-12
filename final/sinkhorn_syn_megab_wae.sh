mkdir -p out

dt=$(date '+%d%m%Y%H%M%S');
echo $dt

rec_lambda=1.0
train_size=10000
exp=syn_2_constant_uniform
nat_resampling=None
wae_lambda=1.0
epoch_num=100
pz=uniform
nat_size=100
for wae_lambda in 1.0 0.1 10.0 0.01 100.0
do
for sinkhorn_iters in 10
do
for sinkhorn_epsilon in 0.1
do
for ot_lambda in 0.0
do
for zdim in 2
do
name="syn_2c_wae_res=${nat_resampling}_exp=${exp}_train_size=${train_size}_rec_lambda=${rec_lambda}_zdim=${zdim}_sinkhorn_iters=${sinkhorn_iters}_sinkhorn_epsilon=${sinkhorn_epsilon}_ot_lambda=${ot_lambda}_wae_lambda=${wae_lambda}_${dt}"
python run.py --pz=${pz} --train_size=${train_size} --sinkhorn_sparse=False --enc_noise=deterministic --name="${name}" --nat_size=${nat_size} --nat_resampling=${nat_resampling} --tag="syn,syn_2c,wae" --exp=${exp} --rec_lambda=${rec_lambda} --wae_lambda=${wae_lambda} --epoch_num=${epoch_num} --ot_lambda=${ot_lambda} --sinkhorn_epsilon=${sinkhorn_epsilon} --sinkhorn_iters=${sinkhorn_iters} --zdim=${zdim} --e_pretrain=False --work_dir=out/sinkhorn_${name}  > out/sinkhorn_${name}.cout 2> out/sinkhorn_${name}.cerr
done
done
done
done
done