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
cmd:option('-dicpath',   '', 'Path to dictionary txt file')
cmd:option('-modelpath', '', 'Path to the model')
cmd:option('-initpath',  '', 'Path to text file with initial secction of review')
cmd:option('-devid', 1,  'GPU to use')
cmd:option('-k', 32, 'guesses to rerank')
cmd:option('-r', 10, 'reward per word')
cmd:option('-g', 0.995, 'reward decay')
cmd:option('-v', 0, 'verbosity level')
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
local model = all['model']
local lut = model.modules[1].modules[2].modules[1]
local rnn = model.modules[2]
local dec = model.modules[7]
model:cuda()
model:remove()
model:evaluate()
collectgarbage()

model = nn.Sequential()
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
local f = assert(io.open(config.initpath, "r"))
local line = f:read("*line")
while line ~= nil do
    local init = {}
    for word in line:gmatch("[^ ]+") do
        table.insert(init, data.getidx(dic, word))
    end
    print('')
    local template = { 
                        torch.CudaTensor(init),
                        torch.CudaTensor({data.getidx(dic, '<end>')})
                     }
                       
    local best = decoders.stochastic_template_lengthbiased_beamsearch(
                    model,
                    rnn,
                    dec,
                    config.k,
                    template,
                    dic,
                    config.r,
                    config.g,
                    config.v)
    for i = 1, #best do
        io.write(dic.idx2word[best[i]] .. ' ')
        if i == #init then
          io.write('<begin_generation> ')
        end
    end
    print('')
    
    ne = ne + 1
    if ne % 10 == 0 then collectgarbage() end
    line = f:read("*line")
end
