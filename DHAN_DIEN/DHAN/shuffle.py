# 
# Copyright (C) 2002-2019 Igor Sysoev
# Copyright (C) 2011,2019 Nginx, Inc.
# Copyright (C) 2010-2019 Alibaba Group Holding Limited
# Copyright (C) 2011-2013 Xiaozhe "chaoslawful" Wang
# Copyright (C) 2011-2013 Zhang "agentzh" Yichun
# Copyright (C) 2011-2013 Weibin Yao
# Copyright (C) 2012-2013 Sogou, Inc.
# Copyright (C) 2012-2013 NetEase, Inc.
# Copyright (C) 2014-2017 Intel, Inc.
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  1. Redistributions of source code must retain the above copyright
#      notice, this list of conditions and the following disclaimer.
#  2. Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#  
#   THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
#   ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#   IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#   ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
#   FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
#   DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
#   OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
#   HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
#   LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
#   OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
#   SUCH DAMAGE.
#  

import os
import sys
import random

import tempfile
from subprocess import call


def main(file, temporary=False):
    tf_os, tpath = tempfile.mkstemp(dir='~/DIN-V2-CODE')
    tf = open(tpath, 'w')

    fd = open(file, "r")
    for l in fd:
        print >> tf, l.strip("\n")
    tf.close()

    lines = open(tpath, 'r').readlines()
    random.shuffle(lines)
    if temporary:
        path, filename = os.path.split(os.path.realpath(file))
        fd = tempfile.TemporaryFile(prefix=filename + '.shuf', dir=path)
    else:
        fd = open(file + '.shuf', 'w')

    for l in lines:
        s = l.strip("\n")
        print >> fd, s

    if temporary:
        fd.seek(0)
    else:
        fd.close()

    os.remove(tpath)

    return fd


if __name__ == '__main__':
    main(sys.argv[1])

