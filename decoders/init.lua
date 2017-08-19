require 'Candidate'

local decoders = {}

decoders.beam_search =
    function(model, rnn, dec, width, template)
        assert(#template) == 2
    
        -- prepare beam
        local prefix = template[1]
        local suffix = template[2]
        local inter = model:forward({rnn:initializeHidden(1), prefix})
        local first_probs, first_idxs = dec:topkseqs(inter, k, prefix)
        local init_hidden = rnn:getLastHidden()
        print(init_hidden:size())
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

decoders.template_beam_search =
    function(model, rnn, width, template)
        error('Template Beam Search has not been implemented!')
    end

decoders.branching_template_beam_search =
    function(model, rnn, width, template, bf)
        error('Branching Template Beam Search has not been implemented!')
    end

return decoders
