import torch, time, pdb, os, random
import numpy as np
import torch.nn as nn
from torch import optim

from freezer import freeze_decoder_weights, check_requires_grad
from utils import *
from nn_layers import *
from parameters import *
import matplotlib.pyplot as plt
import numpy as np


##################### Author @Emre Ozfatura  @ Yulin Shao ###################################################

######################### Inlcluded modules and options #######################################
# 1) Feature extracture
# 2) Successive decoding option
# 3) Vector embedding option
# 4) Belief Modulate

################################# Guideline #####################################
# Current activation is GELU
# trainining for 120000 epoch


################################## Distributed training approach #######################################################

def ModelAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        if w_avg[k].requires_grad: #change1
            for i in range(1, len(w)):
                w_avg[k] += w[i][k]
            w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg


########################## This is the overall AutoEncoder model ########################


class AE(nn.Module):
    def __init__(self, args):
        super(AE, self).__init__()
        self.args = args
        ################## We use learnable positional encoder which can be removed later ######################################
        # self.pe = PositionalEncoder(SeqLen=self.args.K+1, lenWord=args.d_model_trx) # learnable PE
        self.pe = PositionalEncoder_fixed()
        ########################################################################################################################
        if args.embedding == True:
            self.Tmodel = BERT("trx", args.clas + 2 * (args.T - 1), args.m, args.d_model_trx, args.N_trx,
                               args.heads_trx, args.dropout, args.custom_attn, args.multclass, args.NS_model)
        else:
            self.Tmodel = BERT("trx", args.m + 2 * (args.T - 1), args.m, args.d_model_trx, args.N_trx, args.heads_trx,
                               args.dropout, args.custom_attn, args.multclass, args.NS_model)
        ############### ACTIVE FEEDBACK ################
        self.Tmodelfb = BERT("trx", 2 * (args.T - 1) - 1, args.m, args.d_model_trx, args.N_trx, args.heads_trx,
                             args.dropout, args.custom_attn, args.multclass, args.NS_model)

        self.Rmodel1 = BERT("rec", args.T - 2, args.m, args.d_model_trx, args.N_trx + 1, args.heads_trx, args.dropout,
                            args.custom_attn, args.multclass, args.NS_model)
        self.Rmodel2 = BERT("rec", args.T - 1, args.m, args.d_model_trx, args.N_trx + 1, args.heads_trx, args.dropout,
                            args.custom_attn, args.multclass, args.NS_model)
        self.Rmodel3 = BERT("rec", args.T, args.m, args.d_model_trx, args.N_trx + 1, args.heads_trx, args.dropout,
                            args.custom_attn, args.multclass, args.NS_model)
        ########## Power Reallocation as in deepcode work ###############
        if self.args.reloc == 1:
            self.total_power_reloc = Power_reallocate(args)
            self.total_power_reloc_fb = Power_reallocate_fb(args)

    def power_constraint(self, inputs, isTraining, eachbatch, idx=0, fb=0):  # Normalize through batch dimension
        # this_mean = torch.mean(inputs, 0)
        # this_std  = torch.std(inputs, 0)
        if isTraining == 1:
            # training
            this_mean = torch.mean(inputs, 0)
            this_std = torch.std(inputs, 0)
        elif isTraining == 0:
            # test
            statistics_folder = "statistics_snr1_" + str(args.snr1) + "_snr2_" + str(args.snr2) + "_T_" + str(args.T)
            if eachbatch == 0:
                if fb == 0:
                    this_mean = torch.mean(inputs, 0)
                    this_std = torch.std(inputs, 0)
                    if not os.path.exists(statistics_folder):
                        os.mkdir(statistics_folder)
                    torch.save(this_mean, statistics_folder + '/this_mean' + str(idx))
                    torch.save(this_std, statistics_folder + '/this_std' + str(idx))
                    print('this_mean and this_std saved ...')
                else:
                    this_mean = torch.mean(inputs, 0)
                    this_std = torch.std(inputs, 0)
                    if not os.path.exists(statistics_folder):
                        os.mkdir(statistics_folder)
                    torch.save(this_mean, statistics_folder + '/this_mean_fb' + str(idx))
                    torch.save(this_std, statistics_folder + '/this_std_fb' + str(idx))
                    print('this_mean and this_std saved for feedback...')

            else:
                if fb == 0:
                    this_mean = torch.load(statistics_folder + '/this_mean' + str(idx))
                    this_std = torch.load(statistics_folder + '/this_std' + str(idx))
                else:
                    this_mean = torch.load(statistics_folder + '/this_mean_fb' + str(idx))
                    this_std = torch.load(statistics_folder + '/this_std_fb' + str(idx))

        outputs = (inputs - this_mean) * 1.0 / (this_std + 1e-8)
        return outputs

    ########### IMPORTANT ##################
    # We use unmodulated bits at encoder
    #######################################
    def forward(self, eachbatch, bVec_md, fwd_noise_par, fb_noise_par, table=None, isTraining=1):
        ###############################################################################################################################################################
        combined_noise_par = fwd_noise_par + fb_noise_par  # The total noise for parity bits
        outputs = torch.zeros(args.batchSize, args.ell, args.T)
        for idx in range(self.args.T):  # Go through T interactions
            # Generating the parity symbols in the encoder network
            if idx == 0:  # phase 0
                src = torch.cat([bVec_md, torch.zeros(self.args.batchSize, self.args.ell, 2 * (self.args.T - 1)).to(
                    self.args.device)], dim=2)
            elif idx == self.args.T - 1:
                src = torch.cat([bVec_md, parity_all, parity_all_fb_n], dim=2)
            else:
                src = torch.cat([bVec_md, parity_all,
                                 torch.zeros(self.args.batchSize, args.ell, self.args.T - (idx + 1)).to(
                                     self.args.device), parity_all_fb_n,
                                 torch.zeros(self.args.batchSize, args.ell, self.args.T - (idx + 1)).to(
                                     self.args.device)], dim=2)
            ############# Generate the output ###################################################
            output = self.Tmodel(src, None, self.pe)
            parity = self.power_constraint(output, isTraining, eachbatch, idx, fb=0)
            parity = self.total_power_reloc(parity, idx)
            # Saving the generated parity symbols in a tensor
            outputs[:, :, idx] = parity.squeeze()
            if idx == 0:
                parity_fb = parity + combined_noise_par[:, :, idx].unsqueeze(-1)
                parity_all = parity
                parity_all_n = parity + fwd_noise_par[:, :, 0].unsqueeze(-1)  # noisy parity symbols used for q_tilde
                ########################## Since we have three decoders ################################
                received1 = parity + fwd_noise_par[:, :, 0].unsqueeze(-1)
                received2 = parity + fwd_noise_par[:, :, 0].unsqueeze(-1)
                received3 = parity + fwd_noise_par[:, :, 0].unsqueeze(-1)
            else:
                parity_all = torch.cat([parity_all, parity], dim=2)
                parity_all_n = torch.cat([parity_all_n, parity + fwd_noise_par[:, :, idx].unsqueeze(-1)], dim=2)
                if idx < self.args.T - 2:
                    received1 = torch.cat([received1, parity + fwd_noise_par[:, :, idx].unsqueeze(-1)], dim=2)
                if idx < self.args.T - 1:
                    received2 = torch.cat([received2, parity + fwd_noise_par[:, :, idx].unsqueeze(-1)], dim=2)
                received3 = torch.cat([received3, parity + fwd_noise_par[:, :, idx].unsqueeze(-1)], dim=2)

            ## Generating the coded feedback symbols
            if idx == 0:
                srcf = torch.cat([parity_all_n,
                                  torch.zeros(args.batchSize, args.ell, self.args.T - (idx + 2)).to(self.args.device),
                                  torch.zeros(self.args.batchSize, args.ell, self.args.T - (idx + 2)).to(
                                      self.args.device)],
                                 dim=2)
                output_fb = self.Tmodelfb(srcf, None, self.pe)
                parity_fb = self.power_constraint(output_fb, isTraining, eachbatch, idx, fb=1)
                parity_fb = self.total_power_reloc_fb(parity_fb, idx)
                parity_all_fb = parity_fb  # The generated (noiseless) feedback symbols to use in q_tilde
                parity_all_fb_n = parity_fb + fb_noise_par[:, :, 0].unsqueeze(-1)
            elif idx == self.args.T - 2:  # Last iteration for feedback
                srcf = torch.cat([parity_all_n, parity_all_fb], dim=2)
                output_fb = self.Tmodelfb(srcf, None, self.pe)
                parity_fb = self.power_constraint(output_fb, isTraining, eachbatch, idx, fb=1)
                parity_fb = self.total_power_reloc_fb(parity_fb, idx)
                parity_all_fb = torch.cat([parity_all_fb, parity_fb], dim=2)
                parity_all_fb_n = torch.cat([parity_all_fb_n, parity_fb + fb_noise_par[:, :, idx].unsqueeze(-1)], dim=2)
            elif idx < self.args.T - 2:
                srcf = torch.cat([parity_all_n,
                                  torch.zeros(self.args.batchSize, args.ell, self.args.T - (idx + 2)).to(
                                      self.args.device),
                                  parity_all_fb,
                                  torch.zeros(self.args.batchSize, args.ell, self.args.T - (idx + 2)).to(
                                      self.args.device)],
                                 dim=2)
                output_fb = self.Tmodelfb(srcf, None, self.pe)
                parity_fb = self.power_constraint(output_fb, isTraining, eachbatch, idx, fb=1)
                parity_fb = self.total_power_reloc_fb(parity_fb, idx)
                parity_all_fb = torch.cat([parity_all_fb, parity_fb], dim=2)
                parity_all_fb_n = torch.cat([parity_all_fb_n, parity_fb + fb_noise_par[:, :, idx].unsqueeze(-1)], dim=2)
        # ------------------------------------------------------------ receiver
        # print(received.shape)
        decSeq1 = self.Rmodel1(received1, None, self.pe, temperature=args.temperature)  # Decode the sequence
        decSeq2 = self.Rmodel2(received2, None, self.pe, temperature=args.temperature)  # Decode the sequence
        decSeq3 = self.Rmodel3(received3, None, self.pe, temperature=args.temperature)  # Decode the sequence
        return decSeq1, decSeq2, decSeq3, outputs


############################################################################################################################################################################


def train_model(model, last_eachbatch, args):
    print("-->-->-->-->-->-->-->-->-->--> start training ...")
    model.train()
    start = time.time()
    epoch_loss_record = []
    flag = 0
    map_vec = 2 ** (torch.arange(args.m))
    map_vec = torch.flip(map_vec, [0])  # Mapping of blocks to class indices
    ################################### Distance based vector embedding ####################
    A_blocks = torch.tensor([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]],
                            requires_grad=False).float()  # Look up table for blocks
    Embed = torch.zeros(args.clas, args.batchSize, args.ell, args.clas)
    for i in range(args.clas):
        embed = torch.zeros(args.clas)
        for j in range(args.clas):  ###### normalize vector embedding #########
            if args.embed_normalize == True:
                embed[j] = (torch.sum(
                    torch.abs(A_blocks[i, :] - A_blocks[j, :])) - 3 / 2) / 0.866  # normalize embedding
            else:
                embed[j] = torch.sum(torch.abs(A_blocks[i, :] - A_blocks[j, :]))
        Embed[i, :, :, :] = embed.repeat(args.batchSize, args.ell, 1)
    #########################################################################################
    for eachbatch in range(last_eachbatch + 1, args.totalbatch):
        
        #change2 Freeze or unfreeze decoders based on the current batch number
        freeze_decoder_weights(model, eachbatch)    
        #check_requires_grad(model)

        if args.embedding == False:
            # BPSK modulated representations 
            bVec = torch.randint(0, 2, (args.batchSize, args.ell, args.m))
            bVec_md = 2 * bVec - 1
        else:  # vector embedding
            bVec = torch.randint(0, args.clas, (args.batchSize, args.ell, 1))
            bVec_md = torch.zeros((args.batchSize, args.ell, args.clas),
                                  requires_grad=False)  # generated data in terms of distance embeddings
            for i in range(args.clas):
                mask = (bVec == i).long()
                bVec_md = bVec_md + (mask * Embed[i, :, :, :])
        #################################### Generate noise sequence ##################################################
        ###############################################################################################################
        ###############################################################################################################
        ################################### Curriculum learning strategy ##############################################
        if eachbatch < args.core * 20000:
            snr1 = 3 * (1 - eachbatch / (args.core * 20000)) + (eachbatch / (args.core * 20000)) * args.snr1
            snr2 = 100
        elif eachbatch < args.core * 40000:
            snr2 = 100 * (1 - (eachbatch - args.core * 20000) / (args.core * 20000)) + (
                        (eachbatch - args.core * 20000) / (args.core * 20000)) * args.snr2
            snr1 = args.snr1
        else:
            snr2 = args.snr2
            snr1 = args.snr1
        ################################################################################################################
        std1 = 10 ** (-snr1 * 1.0 / 10 / 2)  # forward snr
        std2 = 10 ** (-snr2 * 1.0 / 10 / 2)  # feedback snr
        # Noise values for the parity bits
        fwd_noise_par = torch.normal(0, std=std1, size=(args.batchSize, args.ell, args.T), requires_grad=False)
        fb_noise_par = torch.normal(0, std=std2, size=(args.batchSize, args.ell, args.T), requires_grad=False)
        if args.snr2 == 100:
            fb_noise_par = 0 * fb_noise_par
        if np.mod(eachbatch, args.core) == 0:
            w_locals = []
            w0 = model.state_dict()
            w0 = copy.deepcopy(w0)
        else:
            # Use the common model to have a large batch strategy
            model.load_state_dict(w0)

        # feed into model to get predictions
        preds1, preds2, preds3,_ = model(eachbatch, bVec_md.to(args.device), fwd_noise_par.to(args.device),
                                                fb_noise_par.to(args.device), A_blocks.to(args.device), isTraining=1)

        args.optimizer.zero_grad()
        if args.multclass:
            if args.embedding == False:
                bVec_mc = torch.matmul(bVec, map_vec)
                ys = bVec_mc.long().contiguous().view(-1)
            else:
                ys = bVec.contiguous().view(-1)
        else:
            # expand the labels (bVec) in a batch to a vector, each word in preds should be a 0-1 distribution
            ys = bVec.long().contiguous().view(-1)
        ################## for all predictions ################
        preds1 = preds1.contiguous().view(-1, preds1.size(-1))  # => (Batch*K) x 2^m
        preds1 = torch.log(preds1)
        preds2 = preds2.contiguous().view(-1, preds2.size(-1))  # => (Batch*K) x 2^m
        preds2 = torch.log(preds2)
        preds3 = preds3.contiguous().view(-1, preds3.size(-1))  # => (Batch*K) x 2^m
        preds3 = torch.log(preds3)

        # change 3 Compute loss based on the active decoder
        x = ((eachbatch - 1) / 100) % 3
        if x == 0:
            loss = F.nll_loss(preds1, ys.to(args.device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_th)
            args.optimizer.step()
        elif x == 1:
            loss = F.nll_loss(preds2, ys.to(args.device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_th)
            args.optimizer.step()
        elif x == 2:
            loss = F.nll_loss(preds3, ys.to(args.device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_th)
            args.optimizer.step()

        ########################## This should be binary cross-entropy loss
        #loss.backward()
        ####################### Gradient Clipping optional ###########################
        #torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_th)
        ###############################################################################
        #args.optimizer.step()

        # Save the model
        w1 = model.state_dict()
        w_locals.append(copy.deepcopy(w1))
        ###################### untill core number of iterations are completed ####################
        if np.mod(eachbatch, args.core) != args.core - 1:
            continue
        else:
            ########### When core number of models are obtained #####################
            w2 = ModelAvg(w_locals)  # Average the models
            model.load_state_dict(copy.deepcopy(w2))
            ##################### change the learning rate ##########################
            if args.use_lr_schedule:
                args.scheduler.step()
        ################################ Observe test accuracy ##############################
        with torch.no_grad():
            probs, decodeds = preds3.max(dim=1)  ########## use the higher rate ###############
            succRate = sum(decodeds == ys.to(args.device)) / len(ys)
            print('BAAF_VARv1', 'Idx,lr,snr1,snr2,BS,loss,BER,num=', (
                eachbatch, args.lr, args.snr1, args.snr2, args.batchSize, round(loss.item(), 4),
                round(1 - succRate.item(), 6),
                sum(decodeds != ys.to(args.device)).item()))
        ####################################################################################
        # if np.mod(eachbatch, args.core * 50) == args.core - 1:
        #     epoch_loss_record.append(loss.item())
        #     if not os.path.exists('weights'):
        #         os.mkdir('weights')
        #     torch.save(epoch_loss_record, 'weights/loss')

        if np.mod(eachbatch, args.core * 5000) == args.core - 1:
            if not os.path.exists('weights'):
                os.mkdir('weights')
            saveDir = 'weights/AF_weights_' + 'snr1_' + str(args.snr1) + 'snr2_' + str(args.snr2) \
                      + 'epoch_' + str(eachbatch) + 'T_' + str(args.T)
            save_checkpoint(model=model, optimizer=args.optimizer, scheduler=args.scheduler,
                            eachbatch=eachbatch, checkpoint_path=saveDir)



def EvaluateNets(model, args):
    # # ======================================================= load weights
    model.eval()
    map_vec = 2 ** (torch.arange(args.m))
    map_vec = torch.flip(map_vec, [0])  # Mapping of blocks to class indices
    args.numTestbatch = 10000
    ################################### Distance based vector embedding ####################
    A_blocks = torch.tensor([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]],
                            requires_grad=False).float()  # Look up table for blocks
    Embed = torch.zeros(args.clas, args.batchSize, args.ell, args.clas)
    for i in range(args.clas):
        embed = torch.zeros(args.clas)
        for j in range(args.clas):
            if args.embed_normalize == True:
                embed[j] = (torch.sum(torch.abs(A_blocks[i, :] - A_blocks[j, :])) - 3 / 2) / 0.866
            else:
                embed[j] = torch.sum(torch.abs(A_blocks[i, :] - A_blocks[j, :]))
        Embed[i, :, :, :] = embed.repeat(args.batchSize, args.ell, 1)
    # failbits = torch.zeros(args.K).to(args.device)
    bitErrors = 0
    pktErrors = 0
    packet = 0
    adap = 0
    avg_power = 0
    for eachbatch in range(args.numTestbatch):
        if args.embedding == False:
            # BPSK modulated representations 
            bVec = torch.randint(0, 2, (args.batchSize, args.ell, args.m))
            bVec_md = 2 * bVec - 1
        else:  # vector embedding
            bVec = torch.randint(0, args.clas, (args.batchSize, args.ell, 1))
            bVec_md = torch.zeros((args.batchSize, args.ell, args.clas),
                                  requires_grad=False)  # generated data in terms of distance embeddings
            for i in range(args.clas):
                mask = (bVec == i).long()
                bVec_md = bVec_md + (mask * Embed[i, :, :, :])
        # generate n sequence
        std1 = 10 ** (-args.snr1 * 1.0 / 10 / 2)
        std2 = 10 ** (-args.snr2 * 1.0 / 10 / 2)
        fwd_noise_par = torch.normal(0, std=std1, size=(args.batchSize, args.ell, args.T), requires_grad=False)
        fb_noise_par = torch.normal(0, std=std2, size=(args.batchSize, args.ell, args.T), requires_grad=False)
        if args.snr2 == 100:
            fb_noise_par = 0 * fb_noise_par

        # feed into model to get predictions
        with torch.no_grad():
            preds1, preds2, preds3, parity_symbols = model(eachbatch, bVec_md.to(args.device), fwd_noise_par.to(args.device),
                        fb_noise_par.to(args.device), A_blocks.to(args.device), isTraining=0)

            if args.multclass:
                if args.embedding == False:
                    bVec_mc = torch.matmul(bVec, map_vec)
                    ys = bVec_mc.long().contiguous().view(-1)
                else:
                    ys = bVec.contiguous().view(-1)
            else:
                ys = bVec.long().contiguous().view(-1)
            ############# Decode all ##############################
            preds1 = preds1.contiguous().view(-1, preds1.size(-1))
            probs1, decodeds1 = preds1.max(dim=1)
            # print(decodeds1)
            preds2 = preds2.contiguous().view(-1, preds2.size(-1))
            probs2, decodeds2 = preds2.max(dim=1)
            preds3 = preds3.contiguous().view(-1, preds3.size(-1))
            probs3, decodeds3 = preds3.max(dim=1)
            #########################################################
            probs1 = probs1.contiguous().view(args.batchSize, args.ell)
            probs2 = probs2.contiguous().view(args.batchSize, args.ell)
            probs3 = probs3.contiguous().view(args.batchSize, args.ell)
            ############## check trust on block #####################
            flag1 = torch.sum((probs1 < args.conf_th), dim=1)
            flag2 = torch.sum((probs2 < args.conf_th), dim=1)
            flag3 = torch.sum((probs3 < args.conf_th), dim=1)
            ############## trusted blocks ###########################
            mask1 = (flag1 < 1).long()
            mask1 = mask1.unsqueeze(dim=1)
            mask1 = mask1.repeat(1, args.ell).view(-1)
            mask2 = (flag2 < 1).long()
            mask2 = mask2.unsqueeze(dim=1)
            mask2 = mask2.repeat(1, args.ell).view(-1)
            mask3 = (flag3 < 1).long()
            mask3 = mask3.unsqueeze(dim=1)
            mask3 = mask3.repeat(1, args.ell).view(-1)
            ############# Sent Symbols ############################
            num_iterations = torch.ones(args.batchSize) * args.T
            sent = parity_symbols
            sent[flag1 == 0, :, (args.T - 2):] = 0
            num_iterations[flag1 == 0] = args.T-2
            sent[((flag1 > 0) * flag2) == 0, :, args.T - 1] = 0
            num_iterations[((flag1 > 0) * flag2) == 0] = args.T-1
            power = torch.sum(torch.pow(sent, 2), dim=[1, 2])
            avg_power_each_comm_block = power / num_iterations / args.ell
            avg_power_batch = torch.sum(torch.sum(avg_power_each_comm_block)) / args.batchSize
            avg_power = (avg_power * eachbatch + avg_power_batch) / (eachbatch + 1)

            ############# Decisions ###############################
            decisions1 = (decodeds1 != ys.to(args.device)) * mask1
            decisions2 = (decodeds2 != ys.to(args.device)) * mask2 * (1 - mask1)
            decisions3 = (decodeds3 != ys.to(args.device)) * ((1 - mask2) * (1 - mask1))
            allsum = sum(mask1) + sum(mask2 * (1 - mask1)) + sum((1 - mask2) * (1 - mask1))
            print(allsum.item())
            # print(torch.sum(mask1))
            # print(torch.sum(mask2 * (1-mask1)))
            # print(torch.sum((1-mask2) * (1- mask1)))
            adap += (torch.sum(mask2 * (1 - mask1)) + torch.sum((1 - mask2) * (1 - mask1)))
            print(adap.item())
            packet += (torch.sum(mask1) * (args.T - 2) + torch.sum(mask2 * (1 - mask1)) * (args.T - 1) + torch.sum(
                (1 - mask2) * (1 - mask1)) * args.T) / (args.batchSize * args.ell)
            packets = packet / (eachbatch + 1)
            # bitErrors += decisions.sum()
            # BER = bitErrors / (eachbatch + 1) / args.batchSize / args.ell
            pktErrors += (decisions1.view(args.batchSize, args.ell).sum(1).count_nonzero() + decisions2.view(
                args.batchSize, args.ell).sum(1).count_nonzero() + decisions3.view(args.batchSize, args.ell).sum(
                1).count_nonzero())
            PER = pktErrors / (eachbatch + 1) / args.batchSize
            print('BAAF_VARv1', 'num, PER, errors, packet, avg_power = ', eachbatch,
                  round(PER.item(), 11), pktErrors.item(), packets.item(), avg_power.item())
            if eachbatch % 5 == 0:
                metrics = {
                    'eachbatch': eachbatch,
                    'PER': round(PER.item(), 12),
                    'errors': pktErrors.item(),
                    'packets': packets.item(),
                    'average_power': avg_power.item(),
                }
                log_test_metrics(metrics, "metrics_" + 'snr1_' + str(args.snr1) + 'snr2_' + str(args.snr2) \
                                 + 'T_' + str(args.T) + "confTh_" + str(args.conf_th) \
                                 + "_temperature_" + str(args.temperature) + ".txt")
            if pktErrors > 100:
                metrics = {
                    'eachbatch': eachbatch,
                    'PER': round(PER.item(), 12),
                    'errors': pktErrors.item(),
                    'packets': packets.item(),
                    'average_power': avg_power.item(),
                }
                log_test_metrics(metrics, "metrics_" + 'snr1_' + str(args.snr1) + 'snr2_' + str(args.snr2) \
                                 + 'T_' + str(args.T) + "confTh_" + str(args.conf_th) \
                                 + "_temperature_" + str(args.temperature) + ".txt")
                break


if __name__ == '__main__':
    # ======================================================= parse args
    args = args_parser()
    args.device = args.device if torch.cuda.is_available() else 'cpu'
    ########### path for saving model checkpoints ################################
    # args.saveDir = 'weights/model_weights_' + 'snr1_' + str(args.snr1) + 'snr2_' + str(args.snr2) \
    #                + 'epoch_' + str(args.totalbatch) + 'T_' + str(args.T)
    args.saveDir = 'weights/AF_weights_' + 'snr1_' + str(args.snr1) + 'snr2_' + str(args.snr2) \
                      + 'epoch_' + str(args.totalbatch-101) + 'T_' + str(args.T)
    ################## Model size part ###########################################
    args.d_model_trx = args.heads_trx * args.d_k_trx  # total number of features
    # ======================================================= Initialize the model
    model = AE(args).to(args.device)

    # ======================================================= run
    if args.train == 1:
        if args.opt_method == 'adamW':
            args.optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08,
                                               weight_decay=args.wd, amsgrad=False)
        elif args.opt_method == 'lamb':
            args.optimizer = optim.Lamb(model.parameters(), lr=1e-2, betas=(0.9, 0.999), eps=1e-8, weight_decay=args.wd)
        else:
            args.optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9)
        if args.use_lr_schedule:
            lambda1 = lambda epoch: (1 - epoch / args.totalbatch)
            args.scheduler = torch.optim.lr_scheduler.LambdaLR(args.optimizer, lr_lambda=lambda1)
            ######################## huggingface library ####################################################
            # args.scheduler = get_polynomial_decay_schedule_with_warmup(optimizer=args.optimizer, warmup_steps=1000, num_training_steps=args.totalbatch, power=0.5)

        if os.path.exists(args.saveDir):
            model, args.optimizer, args.scheduler, eachbatch = load_checkpoint(model=model, optimizer=args.optimizer,
                                                                               scheduler=args.scheduler,
                                                                               train=args.train,
                                                                               checkpoint_path=args.saveDir,
                                                                               device=args.device)
        else:
            eachbatch = 0

        train_model(model, eachbatch, args)
    else:
        model = load_checkpoint(model=model, optimizer=None, scheduler=None, checkpoint_path=args.saveDir,
                                train=args.train, device=args.device)
        EvaluateNets(model, args)
