require 'torch'
local cmd = torch.CmdLine('-', '-')
cmd:option('-fp', '', 'path to data file')

local config = cmd:parse(arg)

local f = assert(io.open(config.fp, "r"))
local line = f:read("*line")
while line ~= nil do
    print(line)
    line = f:read("*line")
end
