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
cmd:option('-testpath',    '', 'Path to text file with context words')
cmd:option('-devid', 1,  'GPU to use')
cmd:option('-r',  0, 'reward per word')
cmd:option('-g',  0.5, 'reward decay')
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
local f = assert(io.open(config.testpath, "r"))
local line = f:read("*line")
local ntotal, ncorrect = 0, 0
while line ~= nil do
    local i = 0
    local prefix, ending1, ending2, correctchoice = {}, {}, {}, nil
    for col in line:gmatch("[^\t]+") do 
        if i == 0 then
            for word in col:gmatch("[^ ]+") do
                local idx = data.getidx(dic, word)
                table.insert(prefix, idx)
            end
        elseif i == 1 then
            for word in col:gmatch("[^ ]+") do
                local idx = data.getidx(dic, word)
                table.insert(ending1, idx)
            end
        elseif i == 2 then
            for word in col:gmatch("[^ ]+") do
                local idx = data.getidx(dic, word)
                table.insert(ending2, idx)
            end
        elseif i == 3 then 
            correctchoice = tonumber(col)
        else
          error('Too many columns!')
        end
        i = i + 1
    end
    local cws, cwl = {}, {}
    local option1, option2 = {}, {}
    for _, idx in ipairs(prefix) do
        table.insert(option1, idx)
        table.insert(option2, idx)
         if idx ~= 2 then 
             table.insert(cwl, idx)
             cws[idx] = true
         end
    end
    for _, idx in ipairs(ending1) do
      table.insert(option1, idx)
    end
    for _, idx in ipairs(ending2) do
      table.insert(option2, idx)
    end
    assert((correctchoice == 1) or (correctchoice == 2))
                       
    local choice = decoders.contextual_decider(model2,
                                               rnn,
                                               dec,
                                               option1,
                                               option2,
                                               dic,
                                               config.r,
                                               config.g,
                                               cws,
                                               config.cr)
    
    ntotal = ntotal + 1
    ncorrect = ncorrect + (choice == correctchoice and 1 or 0)
    print(ntotal .. ' ' .. ncorrect .. ' ' .. (100 * (ncorrect / ntotal)))

    ne = ne + 1
    if ne % 10 == 0 then collectgarbage() end
    line = f:read("*line")
end

print(ntotal .. ' ' .. ncorrect .. ' ' .. (100 * (ncorrect / ntotal)))
