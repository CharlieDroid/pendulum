from torch.utils.tensorboard import SummaryWriter

text = """
episode 0 score -156.7 avg score -156.7 critic loss 0.00000 actor loss 0.00000 system loss 0.00000 reward loss 0.00000
...saving checkpoint...
episode 1 score -504.5 avg score -330.6 critic loss 0.00000 actor loss 0.00000 system loss 0.00000 reward loss 0.00000
episode 2 score -1001.7 avg score -554.3 critic loss 0.00000 actor loss 0.00000 system loss 0.00000 reward loss 0.00000
episode 3 score -803.1 avg score -616.5 critic loss 0.00000 actor loss 0.00000 system loss 0.00000 reward loss 0.00000
episode 4 score -314.0 avg score -556.0 critic loss 0.00000 actor loss 0.00000 system loss 0.00000 reward loss 0.00000
episode 5 score -488.7 avg score -544.8 critic loss 0.00000 actor loss 0.00000 system loss 0.00000 reward loss 0.00000
episode 6 score -272.3 avg score -505.9 critic loss 0.00000 actor loss 0.00000 system loss 0.00000 reward loss 0.00000
episode 7 score -885.3 avg score -553.3 critic loss 0.00000 actor loss 0.00000 system loss 0.00000 reward loss 0.00000
episode 8 score -1415.6 avg score -649.1 critic loss 0.00000 actor loss 0.00000 system loss 0.00000 reward loss 0.00000
episode 9 score -1132.0 avg score -697.4 critic loss 0.00000 actor loss 0.00000 system loss 0.00000 reward loss 0.00000
...training all warmup steps...
episode 10 score -6132.0 avg score -1191.4 critic loss 15.48842 actor loss -37.75300 system loss 0.00937 reward loss 0.61433
episode 11 score -3611.6 avg score -1393.1 critic loss 46.52121 actor loss -37.52211 system loss 0.01152 reward loss 1.48413
episode 12 score -881.6 avg score -1353.8 critic loss 19.50531 actor loss -28.98146 system loss 0.01121 reward loss 0.79789
episode 13 score -885.7 avg score -1320.3 critic loss 15.42826 actor loss -22.59860 system loss 0.00935 reward loss 0.68208
episode 14 score -2894.9 avg score -1425.3 critic loss 14.11754 actor loss -15.40422 system loss 0.01377 reward loss 0.55519
episode 15 score -948.9 avg score -1395.5 critic loss 18.82335 actor loss -9.59055 system loss 0.01669 reward loss 0.63658
episode 16 score -890.1 avg score -1365.8 critic loss 16.18842 actor loss -3.11908 system loss 0.01767 reward loss 0.49598
episode 17 score -798.1 avg score -1334.3 critic loss 16.09165 actor loss 0.90936 system loss 0.02113 reward loss 0.49425
episode 18 score -775.4 avg score -1304.8 critic loss 15.41130 actor loss 3.12283 system loss 0.02075 reward loss 0.44600
episode 19 score -658.7 avg score -1272.5 critic loss 14.26160 actor loss 5.54084 system loss 0.02126 reward loss 0.43231
episode 20 score -630.4 avg score -1242.0 critic loss 11.70161 actor loss 7.35569 system loss 0.02187 reward loss 0.37844
episode 21 score -654.8 avg score -1215.3 critic loss 12.01445 actor loss 8.07904 system loss 0.02165 reward loss 0.39045
episode 22 score -677.8 avg score -1191.9 critic loss 12.28240 actor loss 8.05015 system loss 0.02179 reward loss 0.35609
episode 23 score -758.2 avg score -1173.8 critic loss 12.84131 actor loss 7.63526 system loss 0.02248 reward loss 0.31339
episode 24 score -697.3 avg score -1154.8 critic loss 12.33989 actor loss 7.05730 system loss 0.02006 reward loss 0.35878
episode 25 score -707.1 avg score -1137.6 critic loss 11.97060 actor loss 6.53874 system loss 0.01918 reward loss 0.26438
episode 26 score -703.0 avg score -1121.5 critic loss 12.90451 actor loss 7.21642 system loss 0.01747 reward loss 0.28371
episode 27 score -591.3 avg score -1102.5 critic loss 11.47028 actor loss 7.60804 system loss 0.02305 reward loss 0.29793
episode 28 score -583.6 avg score -1084.6 critic loss 12.14433 actor loss 7.93430 system loss 0.02041 reward loss 0.27591
episode 29 score -605.0 avg score -1068.6 critic loss 13.38204 actor loss 8.26267 system loss 0.02099 reward loss 0.25612
episode 30 score -4005.2 avg score -1163.4 critic loss 12.56938 actor loss 8.57221 system loss 0.02002 reward loss 0.23076
episode 31 score -676.9 avg score -1148.2 critic loss 14.57509 actor loss 10.02186 system loss 0.01884 reward loss 0.31162
episode 32 score -607.7 avg score -1131.8 critic loss 11.97631 actor loss 12.88929 system loss 0.01632 reward loss 0.28942
episode 33 score -568.4 avg score -1115.2 critic loss 13.83035 actor loss 15.62057 system loss 0.01670 reward loss 0.26323
episode 34 score -751.6 avg score -1104.8 critic loss 14.10357 actor loss 17.86716 system loss 0.01638 reward loss 0.25499
episode 35 score -484.7 avg score -1087.6 critic loss 13.88019 actor loss 20.43722 system loss 0.01679 reward loss 0.23838
episode 36 score -567.5 avg score -1073.5 critic loss 12.53556 actor loss 22.02790 system loss 0.01645 reward loss 0.21939
episode 37 score -637.9 avg score -1062.1 critic loss 11.14246 actor loss 23.71711 system loss 0.01606 reward loss 0.20996
episode 38 score -647.5 avg score -1051.5 critic loss 12.39225 actor loss 25.76452 system loss 0.01517 reward loss 0.24668
episode 39 score -526.5 avg score -1038.3 critic loss 13.17548 actor loss 28.63681 system loss 0.01337 reward loss 0.24087
episode 40 score -494.3 avg score -1025.1 critic loss 11.96423 actor loss 28.48473 system loss 0.01547 reward loss 0.22641
episode 41 score -707.4 avg score -1017.5 critic loss 11.29819 actor loss 28.98431 system loss 0.01260 reward loss 0.20472
episode 42 score -666.6 avg score -1009.3 critic loss 12.39352 actor loss 27.33122 system loss 0.01553 reward loss 0.22374
episode 43 score -566.0 avg score -999.3 critic loss 10.52621 actor loss 27.85434 system loss 0.01458 reward loss 0.22167
episode 44 score -584.3 avg score -990.0 critic loss 10.40439 actor loss 28.20129 system loss 0.01397 reward loss 0.18317
episode 45 score -548.7 avg score -980.4 critic loss 9.82195 actor loss 29.39408 system loss 0.01465 reward loss 0.19536
episode 46 score -474.8 avg score -969.7 critic loss 8.68255 actor loss 29.72338 system loss 0.01555 reward loss 0.19322
episode 47 score -623.9 avg score -962.5 critic loss 9.56987 actor loss 31.01392 system loss 0.01523 reward loss 0.19618
episode 48 score -573.6 avg score -954.5 critic loss 8.42172 actor loss 31.54074 system loss 0.01674 reward loss 0.17275
episode 49 score -647.4 avg score -948.4 critic loss 8.96802 actor loss 33.01234 system loss 0.01664 reward loss 0.17635
episode 50 score -655.8 avg score -942.7 critic loss 9.75300 actor loss 34.64276 system loss 0.01531 reward loss 0.17114
episode 51 score -641.8 avg score -936.9 critic loss 11.05087 actor loss 35.91984 system loss 0.01529 reward loss 0.19176
episode 52 score -495.4 avg score -928.5 critic loss 11.77892 actor loss 36.05807 system loss 0.01469 reward loss 0.16901
episode 53 score -588.4 avg score -922.2 critic loss 12.93142 actor loss 35.83654 system loss 0.01463 reward loss 0.16093
episode 54 score -577.7 avg score -916.0 critic loss 12.73562 actor loss 36.13789 system loss 0.01462 reward loss 0.15257
episode 55 score -675.1 avg score -911.7 critic loss 11.38263 actor loss 35.95038 system loss 0.01354 reward loss 0.16509
episode 56 score -610.4 avg score -906.4 critic loss 9.36990 actor loss 35.20792 system loss 0.01608 reward loss 0.16900
episode 57 score -502.0 avg score -899.4 critic loss 11.71221 actor loss 37.30374 system loss 0.01314 reward loss 0.11665
episode 58 score -498.2 avg score -892.6 critic loss 9.84332 actor loss 38.67429 system loss 0.01365 reward loss 0.14991
episode 59 score -531.2 avg score -886.6 critic loss 9.84107 actor loss 39.56449 system loss 0.01349 reward loss 0.13626
episode 60 score -459.2 avg score -879.6 critic loss 10.13819 actor loss 40.76865 system loss 0.01420 reward loss 0.13939
episode 61 score -497.2 avg score -873.4 critic loss 10.71330 actor loss 41.76049 system loss 0.01388 reward loss 0.15412
episode 62 score -424.1 avg score -866.3 critic loss 11.66775 actor loss 41.63087 system loss 0.01508 reward loss 0.13010
episode 63 score -479.3 avg score -860.2 critic loss 9.00485 actor loss 43.18532 system loss 0.01295 reward loss 0.15138
episode 64 score -287.9 avg score -851.4 critic loss 9.79901 actor loss 43.56381 system loss 0.01282 reward loss 0.13632
episode 65 score -248.5 avg score -842.3 critic loss 9.39027 actor loss 44.02858 system loss 0.01254 reward loss 0.11190
episode 66 score -328.1 avg score -834.6 critic loss 8.86255 actor loss 43.23716 system loss 0.01434 reward loss 0.12012
episode 67 score -409.0 avg score -828.4 critic loss 7.85789 actor loss 43.27586 system loss 0.01291 reward loss 0.13903
episode 68 score -498.4 avg score -823.6 critic loss 8.29383 actor loss 43.26637 system loss 0.01428 reward loss 0.11248
episode 69 score -465.7 avg score -818.5 critic loss 7.73943 actor loss 43.29971 system loss 0.01535 reward loss 0.14584
episode 70 score -433.6 avg score -813.1 critic loss 7.70776 actor loss 45.31357 system loss 0.01301 reward loss 0.12994
episode 71 score -452.8 avg score -808.1 critic loss 7.41059 actor loss 44.49512 system loss 0.01318 reward loss 0.12199
episode 72 score -304.0 avg score -801.2 critic loss 8.43275 actor loss 44.95081 system loss 0.01226 reward loss 0.11350
episode 73 score -367.6 avg score -795.3 critic loss 7.76765 actor loss 44.70959 system loss 0.01321 reward loss 0.12536
episode 74 score -335.6 avg score -789.2 critic loss 7.60918 actor loss 44.06821 system loss 0.01328 reward loss 0.11441
episode 75 score -355.6 avg score -783.5 critic loss 7.73128 actor loss 43.53479 system loss 0.01359 reward loss 0.11127
episode 76 score -367.3 avg score -778.1 critic loss 7.50098 actor loss 43.78053 system loss 0.01388 reward loss 0.13511
episode 77 score -272.0 avg score -771.6 critic loss 7.55356 actor loss 44.55971 system loss 0.01237 reward loss 0.10110
episode 78 score -243.5 avg score -764.9 critic loss 7.36943 actor loss 43.94981 system loss 0.01417 reward loss 0.11375
episode 79 score -159.2 avg score -757.3 critic loss 7.60029 actor loss 44.92849 system loss 0.01154 reward loss 0.11106
episode 80 score 70.4 avg score -747.1 critic loss 7.37429 actor loss 44.21362 system loss 0.01240 reward loss 0.11304
episode 81 score 293.5 avg score -734.4 critic loss 8.70563 actor loss 43.97594 system loss 0.01305 reward loss 0.10928
episode 82 score -217.5 avg score -728.2 critic loss 9.09463 actor loss 44.88620 system loss 0.01173 reward loss 0.10736
episode 83 score -351.8 avg score -723.7 critic loss 10.63748 actor loss 44.52681 system loss 0.01312 reward loss 0.10780
episode 84 score -582.2 avg score -722.0 critic loss 10.65653 actor loss 43.57504 system loss 0.01271 reward loss 0.12579
episode 85 score -263.1 avg score -716.7 critic loss 9.74356 actor loss 43.40289 system loss 0.01157 reward loss 0.11516
episode 86 score 449.5 avg score -703.3 critic loss 9.84457 actor loss 42.42072 system loss 0.01069 reward loss 0.13262
episode 87 score -332.7 avg score -699.1 critic loss 9.67916 actor loss 41.37344 system loss 0.01105 reward loss 0.12844
episode 88 score -217.9 avg score -693.7 critic loss 9.22784 actor loss 41.43160 system loss 0.01128 reward loss 0.12733
episode 89 score 482.4 avg score -680.6 critic loss 8.87984 actor loss 41.97707 system loss 0.01023 reward loss 0.12590
episode 90 score 128.9 avg score -671.7 critic loss 9.22384 actor loss 42.10473 system loss 0.00974 reward loss 0.14172
episode 91 score -284.1 avg score -667.5 critic loss 8.56873 actor loss 42.15426 system loss 0.01074 reward loss 0.12400
episode 92 score 463.4 avg score -655.3 critic loss 8.80259 actor loss 42.57410 system loss 0.01072 reward loss 0.12686
episode 93 score 826.4 avg score -639.6 critic loss 8.26759 actor loss 41.69855 system loss 0.01252 reward loss 0.11776
episode 94 score -22.3 avg score -633.1 critic loss 7.95729 actor loss 41.95748 system loss 0.01002 reward loss 0.14711
episode 95 score 802.1 avg score -618.1 critic loss 7.81678 actor loss 41.80077 system loss 0.01102 reward loss 0.12837
episode 96 score 673.2 avg score -604.8 critic loss 7.64209 actor loss 41.25128 system loss 0.01098 reward loss 0.12210
episode 97 score 674.0 avg score -591.8 critic loss 7.24372 actor loss 40.63458 system loss 0.01039 reward loss 0.12924
episode 98 score 516.0 avg score -580.6 critic loss 7.53950 actor loss 40.14361 system loss 0.00962 reward loss 0.12374
episode 99 score 804.3 avg score -566.7 critic loss 7.46606 actor loss 39.22291 system loss 0.00952 reward loss 0.11792
episode 100 score 584.9 avg score -559.3 critic loss 8.07151 actor loss 37.67360 system loss 0.00959 reward loss 0.11136
episode 101 score 603.8 avg score -548.2 critic loss 7.73561 actor loss 36.95942 system loss 0.00971 reward loss 0.12667
episode 102 score 865.4 avg score -529.6 critic loss 7.09334 actor loss 35.21812 system loss 0.01060 reward loss 0.11296
episode 103 score 902.0 avg score -512.5 critic loss 8.20034 actor loss 33.67300 system loss 0.01052 reward loss 0.12572
episode 104 score 873.4 avg score -500.6 critic loss 7.62579 actor loss 33.39407 system loss 0.00917 reward loss 0.11687
episode 105 score 857.9 avg score -487.2 critic loss 7.64071 actor loss 31.61968 system loss 0.00997 reward loss 0.11523
episode 106 score 898.2 avg score -475.5 critic loss 7.13884 actor loss 30.20631 system loss 0.01025 reward loss 0.11511
episode 107 score 916.7 avg score -457.4 critic loss 7.47252 actor loss 28.39917 system loss 0.01044 reward loss 0.12197
episode 108 score 874.5 avg score -434.5 critic loss 7.64981 actor loss 27.41378 system loss 0.00941 reward loss 0.10772
episode 109 score 917.7 avg score -414.0 critic loss 7.56580 actor loss 26.30137 system loss 0.00991 reward loss 0.11026
episode 110 score 896.5 avg score -343.8 critic loss 7.17254 actor loss 25.50155 system loss 0.00848 reward loss 0.10744
episode 111 score 841.8 avg score -299.2 critic loss 7.45841 actor loss 24.62828 system loss 0.00890 reward loss 0.11808
episode 112 score 207.9 avg score -288.3 critic loss 7.25802 actor loss 23.20061 system loss 0.00943 reward loss 0.11405
episode 113 score 625.6 avg score -273.2 critic loss 7.47013 actor loss 22.01428 system loss 0.00982 reward loss 0.11367
episode 114 score 801.9 avg score -236.2 critic loss 7.93508 actor loss 20.89622 system loss 0.00902 reward loss 0.10919
episode 115 score 350.6 avg score -223.2 critic loss 7.88283 actor loss 19.52294 system loss 0.00931 reward loss 0.11409
episode 116 score 917.4 avg score -205.2 critic loss 7.49225 actor loss 19.43323 system loss 0.00983 reward loss 0.11147
episode 117 score 844.8 avg score -188.7 critic loss 6.85357 actor loss 18.90561 system loss 0.00904 reward loss 0.10226
episode 118 score 522.2 avg score -175.8 critic loss 7.42963 actor loss 18.45609 system loss 0.00931 reward loss 0.11464
episode 119 score 896.9 avg score -160.2 critic loss 6.35267 actor loss 19.04051 system loss 0.00865 reward loss 0.10817
episode 120 score 776.6 avg score -146.1 critic loss 6.56245 actor loss 17.66100 system loss 0.00965 reward loss 0.09490
...saving checkpoint...
episode 121 score 837.7 avg score -131.2 critic loss 6.84281 actor loss 16.90097 system loss 0.00921 reward loss 0.11649
...saving checkpoint...
episode 122 score 882.9 avg score -115.6 critic loss 7.06764 actor loss 15.79321 system loss 0.00889 reward loss 0.10219
...saving checkpoint...
episode 123 score 767.6 avg score -100.4 critic loss 6.73369 actor loss 15.16616 system loss 0.00921 reward loss 0.10200
...saving checkpoint...
episode 124 score 855.7 avg score -84.8 critic loss 6.80611 actor loss 14.45567 system loss 0.00873 reward loss 0.12825
...saving checkpoint...
episode 125 score -17.2 avg score -77.9 critic loss 7.28124 actor loss 13.40588 system loss 0.00740 reward loss 0.11087
...saving checkpoint...
episode 126 score 892.2 avg score -62.0 critic loss 6.91418 actor loss 12.93424 system loss 0.00797 reward loss 0.10685
...saving checkpoint...
episode 127 score 881.0 avg score -47.2 critic loss 8.31402 actor loss 12.35127 system loss 0.00890 reward loss 0.11379
...saving checkpoint...
episode 128 score 860.1 avg score -32.8 critic loss 7.40750 actor loss 12.42709 system loss 0.00804 reward loss 0.09939
...saving checkpoint...
episode 129 score 811.1 avg score -18.6 critic loss 6.93821 actor loss 11.35010 system loss 0.01008 reward loss 0.10791
...saving checkpoint...
episode 130 score 856.3 avg score 30.0 critic loss 6.83865 actor loss 10.76384 system loss 0.00896 reward loss 0.10983
...saving checkpoint...
episode 131 score 888.6 avg score 45.6 critic loss 6.68365 actor loss 10.32629 system loss 0.00800 reward loss 0.11638
...saving checkpoint...
episode 132 score 889.7 avg score 60.6 critic loss 7.17783 actor loss 9.60106 system loss 0.00947 reward loss 0.09616
...saving checkpoint...
episode 133 score 896.5 avg score 75.2 critic loss 7.02851 actor loss 8.97687 system loss 0.00783 reward loss 0.09504
...saving checkpoint...
episode 134 score 723.8 avg score 90.0 critic loss 6.21852 actor loss 8.02472 system loss 0.00696 reward loss 0.12466
...saving checkpoint...
episode 135 score 897.5 avg score 103.8 critic loss 7.25892 actor loss 7.45063 system loss 0.00925 reward loss 0.10326
...saving checkpoint...
episode 136 score 896.8 avg score 118.5 critic loss 6.88620 actor loss 6.46642 system loss 0.00739 reward loss 0.08927
...saving checkpoint..."""
# make sure no newline above this in the text
text = text.replace("\n...saving checkpoint...", "")
text = text.replace("\n...training all warmup steps...", "").split("\n")[1:]
scores = []
critic_losses = []
actor_losses = []
system_losses = []
reward_losses = []
for line in text:
    line = line.split(" ")
    scores.append(float(line[3]))
    critic_losses.append(float(line[9]))
    actor_losses.append(float(line[12]))
    system_losses.append(float(line[15]))
    reward_losses.append(float(line[18]))
data = [
    (score, critic_loss, actor_loss, system_loss, reward_loss)
    for score, critic_loss, actor_loss, system_loss, reward_loss in zip(
        scores, critic_losses, actor_losses, system_losses, reward_losses
    )
]

filename = "td3 fork retraining 16x16 with system loss and reward loss"
writer = SummaryWriter(log_dir=f"../runs/inverted_pendulum_sim/{filename}")
for i, d in enumerate(data):
    writer.add_scalar("train/reward", d[0], i)
    writer.add_scalar("train/critic_loss", d[1], i)
    writer.add_scalar("train/actor_loss", d[2], i)
    writer.add_scalar("train/system_loss", d[3], i)
    writer.add_scalar("train/reward_loss", d[4], i)
    writer.flush()
writer.close()
