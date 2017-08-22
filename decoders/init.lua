require 'decoders.Candidate'

function shallowcopy(orig)
    local orig_type = type(orig)
    local copy
    if orig_type == 'table' then
        copy = {}
        for orig_key, orig_value in pairs(orig) do
            copy[orig_key] = orig_value
        end
    else -- number, string, boolean, etc
        copy = orig
    end
    return copy
end

local decoders = {}

decoders.beam_search =
    function(model, rnn, dec, width, template)
        assert(#template == 2)
    
        -- prepare beam
        local prefix = template[1]
        local suffix = template[2]
        local inter = model:forward({rnn:initializeHidden(1), prefix})
        local proc_prefix = prefix:resize(prefix:size(1), 1)
        local first_probs, first_idxs = dec:topknext(inter, width, proc_prefix)
        first_idxs:transpose(1, 2)
        first_idxs = first_idxs[2]
        local init_hidden = rnn:getLastHidden()[1]
        local state_vec_len = init_hidden[1]:size(3)
        local term = suffix[1]
        local beam = {}
        for i = 1, width do 
            local t_loc = { chunk_idx=2, word_idx=1, looking_for=suffix[1] }
            local cand = Candidate( {
                                        state = init_hidden,
                                        next_token = first_idxs[i],
                                        t_loc = t_loc
                                  } )
            table.insert(beam, cand)
        end
    
        -- beam search
        local best = nil
        local state = { 
                         torch.CudaTensor(1, width, state_vec_len),
                         torch.CudaTensor(1, width, state_vec_len)
                      }
        local input = torch.CudaTensor(1, width)
        while (not best) or (best.p < beam[1].p) do
            -- prepare state
            for i = 1, width do
                state[1][1][i] = beam[i].state[1]
                state[2][1][i] = beam[i].state[2]
                input[1][i] = beam[i].next_token
            end

            -- step RNN once
            local inter = model:forward({{state}, input})
            local tok_probs, tok_idxs = dec:topknext(inter, width*width, input)
            local nu_hidden = rnn.getLastHidden()[1]

            -- update beam
            local nu_beam = {}
            while #nu_beam < width do
                local i = #nu_beam + 1
                local p, c, w = tok_probs[i], tok_idxs[i][1], tok_idxs[i][2]
                local cand = beam[c]
                local term = cand.t_loc.looking_for
                local nu_seq = table.insert(shallowcopy(cand.seq), w)
                if w == term then
                    if best == nil or p > best.prob then
                        best = Candidate( 
                                          { 
                                            p = p,
                                            seq = nu_seq,
                                            state = {},
											t_loc = {}
                                          }
                                        )
                    end
                else
                    local nu_state = { nu_hidden[1][1][c], nu_hidden[2][1][c] }
                    local nu_cand = Candidate(
                                                {
                                                    p = p,
                                                    next_token = w,
                                                    seq = nu_seq,
                                                    t_loc = cand.t_loc,
                                                    state = nu_state
                                                }
                                             )
                    table.insert(nu_beam, nu_cand)
                end
            end
            beam = nu_beam
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
