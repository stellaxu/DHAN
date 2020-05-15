/* 
 * Copyright (C) 2002-2019 Igor Sysoev
 * Copyright (C) 2011,2019 Nginx, Inc.
 * Copyright (C) 2010-2019 Alibaba Group Holding Limited
 * Copyright (C) 2011-2013 Xiaozhe "chaoslawful" Wang
 * Copyright (C) 2011-2013 Zhang "agentzh" Yichun
 * Copyright (C) 2011-2013 Weibin Yao
 * Copyright (C) 2012-2013 Sogou, Inc.
 * Copyright (C) 2012-2013 NetEase, Inc.
 * Copyright (C) 2014-2017 Intel, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 */

import sys
import hashlib
import random

fin = open("jointed-new-split-info", "r")
ftrain = open("local_train", "w")
ftest = open("local_test", "w")

last_user = "0"
common_fea = ""
line_idx = 0
for line in fin:
    items = line.strip().split("\t")
    ds = items[0]
    clk = int(items[1])
    user = items[2]
    movie_id = items[3]
    dt = items[5]
    cat1 = items[6]

    if ds=="20180118":
        fo = ftrain
    else:
        fo = ftest
    if user != last_user:
        movie_id_list = []
        cate1_list = []
        #print >> fo, items[1] + "\t" + user + "\t" + movie_id + "\t" + cat1 +"\t" + "" + "\t" + "" 
    else:
        history_clk_num = len(movie_id_list)
        cat_str = ""
        mid_str = ""
        for c1 in cate1_list:
            cat_str += c1 + ""
        for mid in movie_id_list:
            mid_str += mid + ""
        if len(cat_str) > 0: cat_str = cat_str[:-1]
        if len(mid_str) > 0: mid_str = mid_str[:-1]
        if history_clk_num >= 1:    # 8 is the average length of user behavior
            print >> fo, items[1] + "\t" + user + "\t" + movie_id + "\t" + cat1 +"\t" + mid_str + "\t" + cat_str
    last_user = user
    if clk:
        movie_id_list.append(movie_id)
        cate1_list.append(cat1)                
    line_idx += 1
