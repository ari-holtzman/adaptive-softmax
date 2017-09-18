require 'math'
require 'cutorch'
require 'nn'
require 'cunn'
require 'rnnlib'

local tablex  = require 'pl.tablex'
local stringx = require 'pl.stringx'
local tnt     = require 'torchnet'
local data    = require 'data'
local utils   = require 'utils'

local decoders = dofile 'decoders/init.lua'

torch.setheaptracking(true)

local cmd = torch.CmdLine('-', '-') cmd:option('-seed', 1, 'random gen seed')
cmd:option('-dicpath',      '', 'Path to dictionary txt file')
cmd:option('-modelpath',    '', 'Path to the model')
cmd:option('-contextpath',    '', 'Path to text file with context words')
cmd:option('-devid', 1,  'GPU to use')
cmd:option('-k', 128, 'guesses to rerank')
cmd:option('-r',  0, 'reward per word')
cmd:option('-g',  0.5, 'reward decay')
cmd:option('-maxsteps', 100, 'reward per word')
cmd:option('-cr',  0, 'reward per context word')
cmd:option('-v',  0, 'verbosity')

local config = cmd:parse(arg)

torch.manualSeed(config.seed)
cutorch.manualSeed(config.seed)
cutorch.setDevice(config.devid)

local dic
if paths.filep(config.dicpath) then
    dic = data.loaddictionary(config.dicpath)
else
    error('Dictionary not found!')
end
local all = torch.load(config.modelpath)
dic = data.sortthresholddictionary(dic, all.config.threshold or 2)
collectgarbage()

local ntoken = #dic.idx2word 
model = all['model']
local lut = model.modules[1].modules[2].modules[1]
local rnn = model.modules[2]
local dec = model.modules[7]
model:cuda()
model:remove()
model:evaluate()
collectgarbage()

local model2 = nn.Sequential()
   :add(nn.ParallelTable()
           :add(nn.Identity())
           :add(nn.Sequential()
                   :add(lut)
                   :add(nn.SplitTable(1))
               )
       )
   :add(rnn)
   :add(nn.SelectTable(2))
   :add(nn.SelectTable(-1))
   :add(nn.JoinTable(1)):cuda()

local ne = 0
local f = assert(io.open(config.contextpath, "r"))
local line = f:read("*line")
while line ~= nil do
    local init_seq = {}
    local i = 0
    for column in line:gmatch("[^\t]+") do
        if i == 0  then
            for word in column:gmatch("[^ ]+") do
                io.write(word .. ' ')
                local idx = data.getidx(dic, word)
                if idx ~= 2 then 
                    table.insert(init_seq, idx)
                end
            end
        else
            break
        end
        i = i + 1
    end
    io.write('* ')
    local term  = data.getidx(dic, '<end>')
    --table.insert(init_seq, data.getidx(dic, '</s>'))
    local prefix = torch.CudaTensor(init_seq)
    local best = decoders.finish_prefix_argmax(model2,
                                              rnn,
                                              dec,
                                              prefix,
                                              term,
                                              dic,
                                              100)
    for i = 1, #best do
        io.write(dic.idx2word[best[i]] .. ' ')
    end
    print('')
    
    ne = ne + 1
    if ne % 10 == 0 then collectgarbage() end
    line = f:read("*line")
end
