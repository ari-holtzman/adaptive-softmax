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
        local init_seq = {}
        for i = 1, prefix:size(1) do table.insert(init_seq, prefix[i]) end
        local proc_prefix = prefix:view(prefix:size(1), 1)
        local suffix = template[2]
        local inter = model:forward({rnn:initializeHidden(1), proc_prefix})
        local base_probs = torch.CudaTensor(1):zero()
        local first_probs, first_idxs = dec:topknext(inter, 
                                                     width+1,
                                                     proc_prefix,
                                                     base_probs)
        first_idxs = first_idxs:t()
        first_idxs = first_idxs[2]
        local raw_hidden = rnn:getLastHidden()
        local state_vec_len = raw_hidden[1][1]:size(3)
        local term = suffix[1]
        local init_hidden = { 
                              raw_hidden[1][1]:clone(),
                              raw_hidden[1][2]:clone()
                            }
        local beam = {}
        for i = 1, width do 
            local t_loc = { chunk_idx=2, word_idx=1, looking_for=suffix[1] }
            local cand = Candidate( {
                                        state = init_hidden,
                                        next_token = first_idxs[i],
                                        t_loc = t_loc,
                                        seq = init_seq
                                  } )
            table.insert(beam, cand)
        end
    
        -- beam search
        local best = nil
        local base_probs = torch.CudaTensor(width):zero()
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
                base_probs[i] = beam[i].p
            end

            -- step RNN once
            local inter = model:forward({{state}, input})
            local tok_probs, tok_idxs = dec:topknext(inter,
                                                    width*width,
                                                    input,
                                                    base_probs)
            local nu_hidden = rnn:getLastHidden()[1]


            -- update beam
            local nu_beam = {}
            local i = 0
            while #nu_beam < width do
                i = i + 1
                local p, c, w = tok_probs[i], tok_idxs[i][1], tok_idxs[i][2]
                local cand = beam[c]
                local term = cand.t_loc.looking_for
                local nu_seq = shallowcopy(cand.seq)
                table.insert(nu_seq, w)
                if w == term then
                    if best == nil or p > best.p then
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
                    local nu_state = { nu_hidden[1][1][c]:clone(),
                                       nu_hidden[2][1][c]:clone()
                                     }
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
    function(model, rnn, dec, width, template)
        local result = {}
        local state = rnn:initializeHidden(1)
        local state_vec_len = state[1][1]:size(3)
        for c = 1, (#template-1) do
            -- prepare beam
            local prefix = template[c]
            local proc_prefix = prefix:view(prefix:size(1), 1)
            for i = 1, prefix:size(1) do table.insert(result, prefix[i]) end
            local inter = model:forward({state, proc_prefix})

            local base_probs = torch.CudaTensor(1):zero()
            local first_probs, first_idxs = dec:topknext(inter, 
                                                         width+1,
                                                         proc_prefix,
                                                         base_probs)
            first_idxs = first_idxs:t()
            first_idxs = first_idxs[2]
            local raw_hidden = rnn:getLastHidden()
            local term = template[c+1][1]
            local init_hidden = { 
                                  raw_hidden[1][1]:clone(),
                                  raw_hidden[1][2]:clone()
                                }
            local init_seq = shallowcopy(result)
            local beam = {}
            local i = 1
            while #beam < width do 
                if first_idxs[i] ~= term then
                    local cand = Candidate( {
                                                state = init_hidden,
                                                next_token = first_idxs[i],
                                                seq = init_seq
                                          } )
                    table.insert(beam, cand)
                end
            end
    
            -- beam search
            local best = nil
            local base_probs = torch.CudaTensor(width):zero()
            local cur_state = { 
                             torch.CudaTensor(1, width, state_vec_len),
                             torch.CudaTensor(1, width, state_vec_len)
                          }
            local input = torch.CudaTensor(1, width)
            while (not best) or (best.p < beam[1].p) do
                print(best)
                -- prepare state
                for i = 1, width do
                    cur_state[1][1][i] = beam[i].state[1]
                    cur_state[2][1][i] = beam[i].state[2]
                    input[1][i] = beam[i].next_token
                    base_probs[i] = beam[i].p
                end

                -- step RNN once
                local inter = model:forward({{cur_state}, input})
                local tok_probs, tok_idxs = dec:topknext(inter,
                                                        width*width,
                                                        input,
                                                        base_probs)
                local nu_hidden = rnn:getLastHidden()[1]

                -- update beam
                local nu_beam = {}
                local i = 0
                while #nu_beam < width do
                    i = i + 1
                    local p, n, w = tok_probs[i], tok_idxs[i][1], tok_idxs[i][2]
                    local cand = beam[n]
                    local nu_seq = shallowcopy(cand.seq)
                    table.insert(nu_seq, w)
                    local nu_state = { 
                                       nu_hidden[1][1][c]:clone(),
                                       nu_hidden[2][1][c]:clone()
                                     }
                    if w == term then
                        if best == nil or p > best.p then
                            best = Candidate( 
                                              { 
                                                p = p,
                                                seq = nu_seq,
                                                state = nu_state
                                              }
                                            )
                        end
                    else
                        local nu_cand = Candidate(
                                                    {
                                                        p = p,
                                                        next_token = w,
                                                        seq = nu_seq,
                                                        state = nu_state
                                                    }
                                                 )
                        table.insert(nu_beam, nu_cand)
                    end
                end
                beam = nu_beam
            end
            state = { best.state }
        end
            
        return result
    end

decoders.branching_template_beam_search =
    function(model, rnn, width, template, bf)
        error('Branching Template Beam Search has not been implemented!')
    end

return decoders