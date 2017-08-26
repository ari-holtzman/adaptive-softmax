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

local cmd = torch.CmdLine('-', '-')
cmd:option('-seed', 1, 'Seed for the random generator')
cmd:option('-modeldir',  '', 'Path to the directory with model and dic')
cmd:option('-textpath',  '', 'Path to text file with sentences')
cmd:option('-devid', 1,  'GPU to use')
cmd:option('-bsz', 128, 'batch size')

local config = cmd:parse(arg)

torch.manualSeed(config.seed)
cutorch.manualSeed(config.seed)
cutorch.setDevice(config.devid)

local modelpath = paths.concat(config.modeldir, 'model.t7')
local dicpath = paths.concat(config.modeldir, 'dic.txt')

local dic
if paths.filep(dicpath) then
    dic = data.loaddictionary(config.dicpath)
else
    error('Dictionary not found!')
end
local all = torch.load(modelpath)
dic = data.sortthresholddictionary(dic, all.config.threshold)
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

local f = assert(io.open(config.textpath, "r"))
local line = f:read("*line")
local ne = 0
local sents = {}
for _, line in pairs(prefix_test_cases) do
    local sent = {}
    local ml = 0
	for word in string.gmatch(line, "[^ ]+") do
    	table.insert(sent, data.getidx(dic, word))
        ml = math.max(#sent, ml)
    end
    table.insert(sents, sent)
    for i = 1, #sents do
        local sent = sents[i] 
        while #sent <  ml do
            table.insert(sent, 0)
        end
    end
    ne = ne + 1
    
    if ne % bsz == 0 then
        local seqs = torch.CudaTensor(sents):t()
        local inter = model2:forward({rnn:initializeHidden(bsz), seqs})
        local probs = dec:getSeqProbs(inter, seqs)
        for i = 1, probs:size(1) do print(probs[i]) end
        sents = {}
    end
    
    if ne % (bsz * 10) == 0 then collectgarbage() end
    line = f:read("*line")
end
