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

local cmd = torch.CmdLine('-', '-') cmd:option('-seed', 1, 'Seed for the random generator')
cmd:option('-dicpath',      '', 'Path to dictionary txt file')
cmd:option('-modelpath',    '', 'Path to the model')
cmd:option('-devid', 1,  'GPU to use')
cmd:option('-k', 128, 'guesses to rerank')
cmd:option('-r',  0, 'reward per word')
cmd:option('-g',  0.5, 'reward decay')
cmd:option('-maxsteps', 100, 'reward per word')

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

local ne = 0
local line = "<beg>\t<end>"
while line ~= nil do
    local template = {}
    local i = 1
    for s in line:gmatch("[^\t]+") do
        local idxs ={}
	    for word in string.gmatch(s, "[^ ]+") do
        	table.insert(idxs, data.getidx(dic, word))
        end
        table.insert(template, torch.CudaTensor(idxs))
        i = i + 1
    end

    local best = decoders.template_beam_search(model2,
                                              rnn,
                                              dec,
                                              config.k,
                                              template,
                                              dic,
                                              config.r,
                                              config.g,
                                              config.maxsteps)
    for i = 1, #best do
        io.write(dic.idx2word[best[i]] .. ' ')
    end
    print('')
    print('---------------------------------')
    
    ne = ne + 1
    if ne % 10 == 0 then collectgarbage() end
    line = io.read()
end
