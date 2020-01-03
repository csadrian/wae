mkdir -p out

dt=$(date '+%d%m%Y%H%M%S');
echo $dt


for seed in 15
do
for bs in 200
do
for lr in 0.0001
do

for z_test in 'sinkhorn'
do
for z_test_scope in 'global'
do
recalculate_size=5000
rec_lambda=1.0
train_size=5000
exp=checkers

for nat_resampling in None
do
mover_ratio=0.5
epoch_num=50
pz=uniform
nat_size=5000
for sinkhorn_iters in 10
do
for sinkhorn_epsilon in  0.01
do
for wae_lambda in 0.1
do
for zdim in 2
do
for stay_lambda in 0.0001
do
name="$checkers_${z_test}_${z_test_scope}_stay_lambda=${stay_lambda}_wae_lambda=${wae_lambda}_${dt}"
python run.py --seed=${seed} --mover_ratio=${mover_ratio} --recalculate_size=${recalculate_size} --pz=${pz} --train_size=${train_size} --z_test=${z_test} --z_test_scope=${z_test_scope} --sinkhorn_sparse=False --enc_noise=deterministic --name="${name}" --nat_size=${nat_size} --nat_resampling=${nat_resampling} --tag="syn,syn_2c,sae" --exp=${exp} --rec_lambda=${rec_lambda} --wae_lambda=${wae_lambda} --epoch_num=${epoch_num} --sinkhorn_epsilon=${sinkhorn_epsilon} --sinkhorn_iters=${sinkhorn_iters} --zdim=${zdim} --e_pretrain=False --batch_size=${bs} --lr=${lr} --stay_lambda=${stay_lambda}  --work_dir=out/sinkhorn_${name}   > out1/sinkhorn_${name}.cout 2> out1/sinkhorn_${name}.cerr
done
done
done
done
done
done
done
done
done
done
done
