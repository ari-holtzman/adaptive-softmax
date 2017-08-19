require 'Candidate'
local decoders = {}
decoders.beam_search =
    function(model, rnn, dec, width, template)
        assert(#template) == 2
        local prefix = template[1]
        local suffix = template[2]
    
        -- run through prefix
        local init_hidden = rnn:initializeHidden(1)
        local inter = model:forward({init_hidden, prefix})
        local opt_probs, _ = dec:topk(inter, config.k, opts:size(1), config.k)
        local first_probs, first_idxs = dec:topk(inter, width, 
        error('Beam Search has not been implemented!') 

        -- prepare beam
        error('Beam Search has not been implemented!') 
        local init_cand = Candidate({ state= }) --finish
        local term = suffix[1]
        local beam = {}
    
        -- beam search
        local best = nil
        while (not best) or (best.p < beam[1].p) do
            error('Beam Search has not been implemented!') 
        end
        
        return best
    end

decoder.template_beam_search =
    function(model, rnn, width, template)
        error('Template Beam Search has not been implemented!')
    end

decoder.branching_template_beam_search =
    function(model, rnn, width, template, bf)
        error('Branching Template Beam Search has not been implemented!')
    end

return decoders
