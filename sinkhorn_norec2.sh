
mkdir -p out

dt=$(date '+%d%m%Y%H%M%S');
echo $dt

rec_lambda=0.0
train_size=1000
exp=celebA
nat_resampling=batch
for sinkhorn_iters in 5 10 20 50
do
for sinkhorn_epsilon in 0.1 0.05 0.2 0.5 0.01 1.0 0.001
do
for ot_lambda in 1.0
do
for zdim in 64
do
name="NOREC_W2_res=${nat_resampling}_exp=${exp}_logdomain_train_size=${train_size}_rec_lambda=${rec_lambda}_zdim=${zdim}__sinkhorn_iters=${sinkhorn_iters}_sinkhorn_epsilon=${sinkhorn_epsilon}_ot_lambda=${ot_lambda}_${dt}"
python run.py --name="${name}" --nat_size=1000 --nat_resampling=${nat_resampling} --tag="NOREC" --exp=${exp} --rec_lambda=${rec_lambda} --train_size ${train_size} --wae_lambda=0.0 --epoch_num=100 --ot_lambda=${ot_lambda} --enc_noise=deterministic --sinkhorn_epsilon=${sinkhorn_epsilon} --sinkhorn_iters=${sinkhorn_iters} --zdim=${zdim} --e_pretrain=False --work_dir=out/sinkhorn_${name}  > out/sinkhorn_${name}.cout 2> out/sinkhorn_${name}.cerr
done
done
done
done