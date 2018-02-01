# RestNet
Rest-Net test Model  From http://ethereon.github.io/netscope/#/gist/db945b393d40bfa26006


        --input image--
         [256 256   3]
        ---------------
              ||
              \/
        ---------------
        -----Conv 1----
         1 128 128 64
        ---BatchNorm---
        -----Scale-----
        -----Relu------
         1 128 128 64
        ---------------
              ||
              \/
        ---------------
        ---Pooling 1---
         1 64 64 64
        ---------------
              ||-----------------------||
              \/                       ||
        ---------------                ||
        ----Conv 21----                || 
          1 64 64 64
        ---BatchNorm---
        -----Scale-----
        -----Relu------                || 
          1 64 64 64
        ---------------
              ||
              \/
        ---------------
        ----Conv 22----                || 
          1 64 64 64
        ---BatchNorm---
        -----Scale-----
        -----Relu------                || 
          1 64 64 64
        ---------------
              ||
              \/
        ---------------
        ----Conv 23----          ----Conv 11----
        1 64 64 256                1 64 64 256
        ---BatchNorm---
        -----Scale-----
              ||<<<-------------------||
              \/
        --------------- 
        ------Res1-----
         1 64 64 256
        ------Relu-----
         1 64 64 256
        ---------------
              ||-----------------------||
              \/                       ||
        ---------------                ||
        ----Conv 31----                || 
          1 32 32 128
        ---BatchNorm---
        -----Scale-----
        -----Relu------                || 
          1 32 32 128
        ---------------
              ||
              \/
        ---------------
        ----Conv 32----                || 
          1 32 32 128
        ---BatchNorm---
        -----Scale-----
        -----Relu------                || 
          1 32 32 128
        ---------------
              ||
              \/
        ---------------
        ----Conv 23----          ----Conv 11----
        1 64 64 512                1 64 64 256
        ---BatchNorm---
        -----Scale-----
              ||<<<-------------------||
              \/
