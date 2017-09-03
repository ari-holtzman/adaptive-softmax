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
local f = assert(io.open(config.contextpath, "r"))
local line = f:read("*line")
while line ~= nil do
    local cws, cwl = {}, {}
    for column in line:gmatch("[^\t]+") do
        for word in column:gmatch("[^ ]+") do
            local idx = data.getidx(dic, word)
            if idx ~= 2 then 
                table.insert(cwl, idx)
                cws[idx] = true
                io.write(dic.idx2word[idx] .. ' ')
            end
        end

        break
    end
    print('')
    local template = { 
                        torch.CudaTensor(cwl),
                        torch.CudaTensor({data.getidx(dic, '</s>')})
                     }
                       
    local best = decoders.template_beam_search(model2,
                                              rnn,
                                              dec,
                                              config.k,
                                              template,
                                              dic)
    for i = 1, #best do
        io.write(dic.idx2word[best[i]] .. ' ')
    end
    print('')
    print('---------------------------------')
    
    ne = ne + 1
    if ne % 10 == 0 then collectgarbage() end
    line = f:read("*line")
end
