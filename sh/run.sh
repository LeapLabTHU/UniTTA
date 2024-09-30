#!/bin/bash

for dataset in {cifar10,cifar100,imagenet,}; do

	adapter_config_file_list=()

	for adapter in {bn,cotta,lame,note,roid,rotta_rbn,rotta,tent,test,tribe,tribe_bbn,unmixtns,unitta_cofa,unitta_bdn,unitta,unitta_cofa_no_filt,unitta_bdn_no_filt,}; do
		adapter_config_file_list="${adapter_config_file_list} configs/adapter/${dataset}/${adapter}.yaml"
	done

	for cor_class_char in {i,n,}; do
		for imb_class_char in {1,u,}; do
			for cor_domain_char in {i,n,1,}; do
				for imb_domain_char in {1,u,}; do

					#
					test_setting=${cor_class_char}${imb_class_char}${cor_domain_char}${imb_domain_char}

					protocol=unitta

					#wandb_mode=online
					#wandb_mode=offline
					wandb_mode=disabled
					wandb_name=unitta

					if [ ${dataset} == "cifar100" ]; then

						unitta_num_domains=15
						unitta_num_classes=100
						unitta_max_state_samples=100

					elif [ ${dataset} == "cifar10" ]; then

						unitta_num_domains=15
						unitta_num_classes=10
						unitta_max_state_samples=1000

					elif [ ${dataset} == "imagenet" ] || [ ${dataset} == "imagenet_vit" ]; then

						unitta_num_domains=15
						unitta_num_classes=1000
						if [ ${cor_class_char} == "n" ]; then
							if [ ${cor_domain_char} == "n" ]; then
								unitta_max_state_samples=100
							elif [ ${cor_domain_char} == "i" ] && [ ${imb_domain_char} == "1" ]; then
								unitta_max_state_samples=50
							else
								unitta_max_state_samples=10
							fi
						else
							unitta_max_state_samples=10
						fi

					fi

					if [ ${cor_domain_char} == "i" ]; then
						unitta_cor_factor_max_domain=0.067
					elif [ ${cor_domain_char} == "n" ]; then
						unitta_cor_factor_max_domain=0.85
					elif [ ${cor_domain_char} == "1" ]; then
						unitta_cor_factor_max_domain=1.0
					else
						echo "Invalid cor_domain_char"
						exit 1
					fi
					if [ ${imb_domain_char} == "1" ]; then
						unitta_imb_factor_domain=1.0
					elif [ ${imb_domain_char} == "u" ]; then
						unitta_imb_factor_domain=5.0
					else
						echo "Invalid imb_domain_char"
						exit 1
					fi

					if [ ${cor_class_char} == "i" ]; then
						unitta_cor_factor_max_class=$(echo "scale=3; 1/${unitta_num_classes}" | bc)
					elif [ ${cor_class_char} == "n" ]; then
						unitta_cor_factor_max_class=0.95
					elif [ ${cor_class_char} == "1" ]; then
						unitta_cor_factor_max_class=1.0
					else
						echo "Invalid cor_class_char"
						exit 1
					fi

					if [ ${imb_class_char} == "1" ]; then
						unitta_imb_factor_class=1.0
					elif [ ${imb_class_char} == "u" ]; then
						unitta_imb_factor_class=10.0
					else
						echo "Invalid imb_class_char"
						exit 1
					fi

					mark=${test_setting}_${dataset}_wandb_${wandb_mode}

					echo ${mark}

					python TTA.py \
						-acfg ${adapter_config_file_list} \
						-dcfg configs/dataset/${dataset}.yaml \
						-pcfg configs/protocol/${protocol}.yaml \
						WANDB.MODE ${wandb_mode} \
						MARK ${mark} \
						WANDB.NAME ${wandb_name} \
						LOADER.SAMPLER.UNITTA_IMB_FACTOR_DOMAIN ${unitta_imb_factor_domain} \
						LOADER.SAMPLER.UNITTA_IMB_FACTOR_CLASS ${unitta_imb_factor_class} \
						LOADER.SAMPLER.UNITTA_COR_FACTOR_MAX_DOMAIN ${unitta_cor_factor_max_domain} \
						LOADER.SAMPLER.UNITTA_COR_FACTOR_MAX_CLASS ${unitta_cor_factor_max_class} \
						LOADER.SAMPLER.UNITTA_NUM_DOMAINS ${unitta_num_domains} \
						LOADER.SAMPLER.UNITTA_NUM_CLASSES ${unitta_num_classes} \
						LOADER.SAMPLER.UNITTA_MAX_STATE_SAMPLES ${unitta_max_state_samples}

				done
			done
		done
	done
done
