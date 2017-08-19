require 'Candidate'
local decoders = {}

decoders.beam_search = function(model, rnn, width, prefix)
    assert(prefix:dim() == 2)

    local sizes = prefix:size()
    local len, bsz = sizes[1], sizes[2]

    local init_hidden = rnn:
    local beam = {}
    
    error('Beam Search has not been implemented!') 
    return decode
end

decoder.template_beam_search = function(model, rnn, width, template)
    error('Template Beam Search has not been implemented!')
end

decoder.branching_template_beam_search = function(model, rnn, width, template, bf)
    error('Branching Template Beam Search has not been implemented!')
end

return decoders
