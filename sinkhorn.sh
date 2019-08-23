
rec_lambda=1.0
train_size=5000
exp=celebA
for sinkhorn_iters in 10
do
for sinkhorn_epsilon in 0.1
do
for ot_lambda in 1000.0
do
for zdim in 64
do
name="longer_marg4max_dd_exp=${exp}_logdomain_train_size=${train_size}_rec_lambda=${rec_lambda}_zdim=${zdim}__sinkhorn_iters=${sinkhorn_iters}_sinkhorn_epsilon=${sinkhorn_epsilon}_ot_lambda=${ot_lambda}"
python run.py --exp=${exp} --rec_lambda=${rec_lambda} --train_size ${train_size} --wae_lambda=0.0 --epoch_num=300 --ot_lambda=${ot_lambda} --enc_noise=deterministic --sinkhorn_epsilon=${sinkhorn_epsilon} --sinkhorn_iters=${sinkhorn_iters} --zdim=${zdim} --e_pretrain=False --work_dir=sinkhorn_${name}  > sinkhorn_${name}.cout 2> sinkhorn_${name}.cerr
done
done
done
done