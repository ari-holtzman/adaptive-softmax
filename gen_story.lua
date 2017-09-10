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
cmd:option('-devid', 1,  'GPU to use')
cmd:option('-k', 128, 'guesses to rerank')
cmd:option('-r',  0, 'reward per word')
cmd:option('-g',  0.5, 'reward decay')
cmd:option('-maxsteps', 100, 'reward per word')
cmd:option('-cr',  0, 'reward per context word')

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
io.write('>'); io.flush()
local line = 'it was a dark and stormy night . '--io.read()
local beg = {}
for word in line:gmatch("[^ ]+") do
    local idx = data.getidx(dic, word)
    table.insert(beg, idx)
end
local state = rnn:initializeHidden(1)
local term = data.getidx(dic, '</s>')

while true do
     local best
     best, state = decoders.beam_search_wstate(
                                model2,
                                rnn,
                                state,
                                dec,
                                config.k,
                                beg,
                                term,
                                dic)

    beg = torch.CudaTensor({term})

    print(best)
    for i = 1, #best do
        io.write(dic.idx2word[best[i]] .. ' ')
    end
    print('')
    print('---------------------------------')
    
    ne = ne + 1
    if ne % 10 == 0 then collectgarbage() end
end
