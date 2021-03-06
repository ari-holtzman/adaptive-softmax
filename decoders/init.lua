require 'decoders.Candidate'

function shallowcopy(orig)
    local orig_type = type(orig)
    local copy
    if orig_type == 'table' then copy = {}
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

decoders.beam_search_wstate = 
    function(model, rnn, init_state, dec, width, init_seq, term)
        -- prepare beam
        local proc_prefix = torch.CudaTensor(init_seq):view(-1, 1)
        local inter = model:forward({init_state, proc_prefix})
        local base_probs = torch.CudaTensor(1):zero()
        local first_probs, first_idxs = dec:topknext(inter, 
                                                     width+1,
                                                     proc_prefix,
                                                     base_probs)
        first_idxs = first_idxs:t()
        first_idxs = first_idxs[2]
        local raw_hidden = rnn:getLastHidden()
        local state_vec_len = raw_hidden[1][1]:size(3)
        local init_hidden = { 
                              raw_hidden[1][1]:clone(),
                              raw_hidden[1][2]:clone()
                            }
        local beam = {}
        for i = 1, width do 
            local t_loc = { chunk_idx=2, word_idx=1, looking_for=term }
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
                    local nu_state = { nu_hidden[1][1][c]:clone():resize(1, 1, state_vec_len),
                                       nu_hidden[2][1][c]:clone():resize(1, 1, state_vec_len)
                                     }
                    if best == nil or p > best.p then
                        best = Candidate( 
                                          { 
                                            p = p,
                                            seq = nu_seq,
                                            state = nu_state,
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
        
        return best.seq, {best.state}
    end


decoders.beam_fill =
    function(model, rnn, dec, width, template)
        error('Not implemented yet') 
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

decoders.contextual_beam_search =
    function(model, rnn, dec, width, template, dic, r, g, max_steps, cws, cr, ll, v)
        local prefix = template[1]
        local inter
        local init_len = template[1]:size(1)
        for c = 1, (#template-1) do
            -- prepare beam
            local state = rnn:initializeHidden(1)
            local state_vec_len = state[1][1]:size(3)
            local proc_prefix = prefix:view(prefix:size(1), 1)
            local inter = model:forward({state, proc_prefix})

            local base_rewards = torch.CudaTensor(1):zero()
            local first_rewards, first_idxs = dec:topknextfunky(inter, 
                                                         width+1,
                                                         proc_prefix,
                                                         base_rewards,
                                                         {cws},
                                                         cr)
            first_idxs = first_idxs:t()
            first_idxs = first_idxs[2]
            local raw_hidden = rnn:getLastHidden()
            local suffix = template[c+1]
            local term = suffix[1]
            local init_hidden = { 
                                  raw_hidden[1][1]:clone(),
                                  raw_hidden[1][2]:clone()
                                }
            local init_seq = {}
            for i = 1, prefix:size(1) do table.insert(init_seq, prefix[i]) end
            local beam = {}
            local i = 1
            while #beam < width do 
                local w = first_idxs[i]
                if w ~= term then
                    local seq = shallowcopy(init_seq)
                    table.insert(seq, w)
                    local nu_cws = shallowcopy(cws)
                    nu_cws[w] = nil
                    local cand = Candidate( {
                                                cws = nu_cws,
                                                state = init_hidden,
                                                next_token = first_idxs[i],
                                                seq = seq,
                                                r =  first_rewards[i]
                                          } )
                    table.insert(beam, cand)
                end
                i = i + 1
            end
    
            -- beam search
            local best = nil
            local base_rewards = torch.CudaTensor(width):zero()
            local input = torch.CudaTensor(1, width)
            local steps = 0
            while steps < max_steps do
                steps = steps + 1
                if v > 0 then
                    for i = 1, #beam[1].seq do
                        io.write(dic.idx2word[beam[1].seq[i]] .. ' ')
                    end
                    print('')
                    
                end

                -- prepare state
                local cur_state = { 
                                    torch.CudaTensor(1, width, state_vec_len),
                                    torch.CudaTensor(1, width, state_vec_len)
                                  }
                local cwss = {}
                for i = 1, width do
                    cur_state[1][1][i] = beam[i].state[1]
                    cur_state[2][1][i] = beam[i].state[2]
                    input[1][i] = beam[i].next_token
                    base_rewards[i] = beam[i].r
                    table.insert(cwss, beam[i].cws)
                end

                -- step RNN once
                local inter = model:forward({{cur_state}, input})
                local tok_rewards, tok_idxs = dec:topknextfunky(inter,
                                                        math.max(width*(width+1), width*2+1),
                                                        input,
                                                        base_rewards,
                                                        cwss,
                                                        cr)
                local nu_hidden = rnn:getLastHidden()[1]

                -- update beam
                local nu_beam = {}
                local i = 0
                while #nu_beam < width do
                    i = i + 1
                    local cur_r, n, w = tok_rewards[i], tok_idxs[i][1], tok_idxs[i][2]
                    local cand = beam[n]
                    local rep = false
                    if ll > 0  and #cand.seq >= ll  then
                        for pos = 1, #cand.seq - ll + 1 do
                          local cur_rep =  true
                          for off = 0, ll-2 do
                            cur_rep = cur_rep and (cand.seq[pos+off] == cand.seq[#cand.seq-ll+off+2])
                          end
                          cur_rep = cur_rep and (cand.seq[pos+ll-1] == w)
                          rep = rep or cur_rep
                          if rep then break end
                        end
                    end
                     if not rep then
                        local nu_seq = shallowcopy(cand.seq)
                        table.insert(nu_seq, w)
                        local nu_r = cur_r 
                        nu_r = nu_r + (r * math.pow(g, #nu_seq-init_len))
                        local nu_state = { 
                                           nu_hidden[1][1][n]:clone(),
                                           nu_hidden[2][1][n]:clone()
                                         }
                        
                        local nu_cws = shallowcopy(cand.cws)
                        nu_cws[w] = nil
                        if w == term then
                            if best == nil or nu_r > best.r then
                                for i = 2, suffix:size(1) do
                                    table.insert(nu_seq, suffix[i])
                                end
                                best = Candidate( 
                                                  { 
                                                    cws = nu_cws,
                                                    r = nu_r, 
                                                    seq = nu_seq,
                                                    state = nu_state
                                                  }
                                                )
                            end
                        else
                            local nu_cand = Candidate(
                                                        {
                                                            r = nu_r,
                                                            next_token = w,
                                                            seq = nu_seq,
                                                            state = nu_state,
                                                            cws = nu_cws
                                                        }
                                                     )
                            table.insert(nu_beam, nu_cand)
                        end
                    end
                end
                beam = nu_beam
            end
            result = best.seq
            prefix = torch.CudaTensor(best.seq)
            state = { best.state }
        end
            
        return result
    end

decoders.contextual_decider =
    function(model, rnn, dec, option1, option2, dic, r, g, cws, cr)
        local state = rnn:initializeHidden(1)
        local t1, t2 = torch.CudaTensor(option1):view(-1, 1), torch.CudaTensor(option2):view(-1, 1)
        local inter = model:forward({state, t1})
        local base1 = dec:getSeqProbs(inter, t1)[1]
        local state = rnn:initializeHidden(1)
        local inter = model:forward({state, t2})
        local base2 = dec:getSeqProbs(inter, t2)[1]

        local lenr1 =  r * ((1 - math.pow(g, #option1)) / (1 - g))
        local lenr2 =  r * ((1 - math.pow(g, #option2)) / (1 - g))

        local contr1, contr2 = 0, 0
        local ws1, ws2 = {}, {}
        for _, idx in pairs(option1) do
            ws1[idx] = true
        end
        for _, idx in pairs(option2) do
            ws2[idx] = true
        end
        for _, idx in pairs(ws1) do
            if cws[idx] then 
              contr1 = contr1 + 1
            end
        end
        for _, idx in pairs(ws2) do
            if cws[idx] then 
              contr2 = contr2 + 1
            end
        end

        local s1, s2 = base1+lenr1+contr1, base2+lenr2+contr2
        local choice = s1 > s2 and 1 or 2

        return choice
    end

decoders.simple_decider =
    function(model, rnn, dec, option1, option2, dic, r, g, cws, cr)
        local state = rnn:initializeHidden(1)
        local t1, t2 = torch.CudaTensor(option1):view(-1, 1), torch.CudaTensor(option2):view(-1, 1)
        local inter = model:forward({state, t1})
        local base1 = dec:getSeqProbs(inter, t1)[1]
        local state = rnn:initializeHidden(1)
        local inter = model:forward({state, t2})
        local base2 = dec:getSeqProbs(inter, t2)[1]

        local s1, s2 = base1, base2
        local choice = s1 > s2 and 1 or 2

        return choice
    end


decoders.template_beam_search =
    function(model, rnn, dec, width, template, dic)
        local prefix = template[1]
        local inter
        for c = 1, (#template-1) do
            -- prepare beam
            local state = rnn:initializeHidden(1)
            local state_vec_len = state[1][1]:size(3)
            local proc_prefix = prefix:view(prefix:size(1), 1)
            local inter = model:forward({state, proc_prefix})

            local base_probs = torch.CudaTensor(1):zero()
            local first_probs, first_idxs = dec:topknext(inter, 
                                                         width+1,
                                                         proc_prefix,
                                                         base_probs)
            first_idxs = first_idxs:t()
            first_idxs = first_idxs[2]
            local raw_hidden = rnn:getLastHidden()
            local suffix = template[c+1]
            local term = suffix[1]
            local init_hidden = { 
                                  raw_hidden[1][1]:clone(),
                                  raw_hidden[1][2]:clone()
                                }
            local init_seq = {}
            for i = 1, prefix:size(1) do table.insert(init_seq, prefix[i]) end
            local beam = {}
            local i = 1
            while #beam < width do 
                local w = first_idxs[i]
                if w ~= term then
                    local seq = shallowcopy(init_seq)
                    table.insert(seq, w)
                    local cand = Candidate( {
                                                state = init_hidden,
                                                next_token = first_idxs[i],
                                                seq = seq
                                          } )
                    table.insert(beam, cand)
                end
                i = i + 1
            end
    
            -- beam search
            local best = nil
            local base_probs = torch.CudaTensor(width):zero()
            local input = torch.CudaTensor(1, width)
            while (not best) or (best.p < beam[1].p) do
                -- prepare state
                local cur_state = { 
                                    torch.CudaTensor(1, width, state_vec_len),
                                    torch.CudaTensor(1, width, state_vec_len)
                                  }
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

                local next_ones = {}
                for i = 1, tok_probs:size(1) do
                    table.insert(next_ones, { p=tok_probs[i], n=tok_idxs[i][1], w=tok_idxs[i][2], f=dic.idx2freq[tok_idxs[i][2]] })
                end
                --table.sort(next_ones,  function (a, b) return a.f < b.f end)

                -- update beam
                local nu_beam = {}
                local i = 0
                while #nu_beam < width do
                    i = i + 1
                    local info = next_ones[i]
                    local p, n, w = info.p, info.n, info.w
                    local cand = beam[n]
                    local nu_seq = shallowcopy(cand.seq)
                    table.insert(nu_seq, w)
                    local nu_state = { 
                                       nu_hidden[1][1][n]:clone(),
                                       nu_hidden[2][1][n]:clone()
                                     }
                    if w == term then
                        if best == nil or p > best.p then
                            for i = 2, suffix:size(1) do
                                table.insert(nu_seq, suffix[i])
                            end
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
            result = best.seq
            prefix = torch.CudaTensor(best.seq)
            state = { best.state }
        end
            
        return result
    end

decoders.stochastic_template_beam_search = function(model, rnn, dec, width, template, dic)
    local prefix = template[1]
    local inter
    for c = 1, (#template-1) do
        -- prepare beam
        local state = rnn:initializeHidden(1)
        local state_vec_len = state[1][1]:size(3)
        local proc_prefix = prefix:view(prefix:size(1), 1)
        local inter = model:forward({state, proc_prefix})

        local base_probs = torch.CudaTensor(1):zero()
        local first_probs, first_idxs = dec:sampleknext(inter, 
                                                     width+1,
                                                     proc_prefix,
                                                     base_probs)
        first_idxs = first_idxs:t()
        first_idxs = first_idxs[2]
        local raw_hidden = rnn:getLastHidden()
        local suffix = template[c+1]
        local term = suffix[1]
        local init_hidden = { 
                              raw_hidden[1][1]:clone(),
                              raw_hidden[1][2]:clone()
                            }
        local init_seq = {}
        for i = 1, prefix:size(1) do table.insert(init_seq, prefix[i]) end
        local beam = {}
        local i = 1
        while #beam < width do 
                             
            local w = first_idxs[i]
            if w ~= term then
                local seq = shallowcopy(init_seq)
                table.insert(seq, w)
                local cand = Candidate( {
                                            state = init_hidden,
                                            next_token = first_idxs[i],
                                            seq = seq
                                      } )
                table.insert(beam, cand)
            end
            i = i + 1
        end

        -- beam search
        local best = nil
        local base_probs = torch.CudaTensor(width):zero()
        local input = torch.CudaTensor(1, width)
        while (not best) or (best.p < beam[1].p) do
            for i = 1, #beam[1].seq do
                io.write(dic.idx2word[beam[1].seq[i]] .. ' ')
            end
            print('')

            -- prepare state
            local cur_state = { 
                                torch.CudaTensor(1, width, state_vec_len),
                                torch.CudaTensor(1, width, state_vec_len)
                              }
            for i = 1, width do
                cur_state[1][1][i] = beam[i].state[1]
                cur_state[2][1][i] = beam[i].state[2]
                input[1][i] = beam[i].next_token
                base_probs[i] = beam[i].p
            end

            -- step RNN once
            local inter = model:forward({{cur_state}, input})
            local tok_probs, tok_idxs = dec:sampleknext(inter,
                                                    width,
                                                    input,
                                                    base_probs)
            local nu_hidden = rnn:getLastHidden()[1]

            local next_ones = {}
            for i = 1, tok_probs:size(1) do
                table.insert(next_ones, { p=tok_probs[i], n=tok_idxs[i][1], w=tok_idxs[i][2], f=dic.idx2freq[tok_idxs[i][2]] })
            end
            --table.sort(next_ones,  function (a, b) return a.f < b.f end)

            -- update beam
            local nu_beam = {}
            local i = 0
            while #nu_beam < width do
                i = i + 1
                local info = next_ones[i]
                local p, n, w = info.p, info.n, info.w
                local cand = beam[n]
                local nu_seq = shallowcopy(cand.seq)
                table.insert(nu_seq, w)
                local nu_state = { 
                                   nu_hidden[1][1][n]:clone(),
                                   nu_hidden[2][1][n]:clone()
                                 }
                if w == term then
                    if best == nil or p > best.p then
                        for i = 2, suffix:size(1) do
                            table.insert(nu_seq, suffix[i])
                        end
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
        result = best.seq
        prefix = torch.CudaTensor(best.seq)
        state = { best.state }
    end
        
    return result
end


decoders.finish_prefix_argmax =
    function(model, rnn, dec, prefix, term, dic, max)
        local result = {}
        local last = {}
        for i = 1, prefix:size(1) do table.insert(result, prefix[i]) end
        local proc_prefix = prefix:view(prefix:size(1), 1)
        local inter = model:forward({rnn:initializeHidden(1), proc_prefix})
        local v = #dic.idx2word
        local idx = dec:nextbest(inter, v)
        while idx ~= term do
            table.insert(result, idx)
            table.insert(last, idx)
            inter = model:forward({rnn:getLastHidden(), torch.CudaTensor({idx})})
            idx = dec:nextbest(inter, v)
            if #last >= max then break end
        end
        table.insert(last, term)
        return last
    end

decoders.branching_template_beam_search =
    function(model, rnn, width, template, bf)
        error('Branching Template Beam Search has not been implemented!')
    end

decoders.length_biased_beam_search =
    function(model, rnn, dec, width, template, dic, r, g, max_steps)
        local prefix = template[1]
        local inter
        local init_len = template[1]:size(1)
        for c = 1, (#template-1) do
            -- prepare beam
            local state = rnn:initializeHidden(1)
            local state_vec_len = state[1][1]:size(3)
            local proc_prefix = prefix:view(prefix:size(1), 1)
            local inter = model:forward({state, proc_prefix})

            local base_rewards = torch.CudaTensor(1):zero()
            local first_rewards, first_idxs = dec:topknext(inter, 
                                                         width+1,
                                                         proc_prefix,
                                                         base_rewards)
            first_idxs = first_idxs:t()
            first_idxs = first_idxs[2]
            local raw_hidden = rnn:getLastHidden()
            local suffix = template[c+1]
            local term = suffix[1]
            local init_hidden = { 
                                  raw_hidden[1][1]:clone(),
                                  raw_hidden[1][2]:clone()
                                }
            local init_seq = {}
            for i = 1, prefix:size(1) do table.insert(init_seq, prefix[i]) end
            local beam = {}
            local i = 1
            while #beam < width do 
                local w = first_idxs[i]
                if w ~= term then
                    local seq = shallowcopy(init_seq)
                    table.insert(seq, w)
                    local cand = Candidate( {
                                                state = init_hidden,
                                                next_token = first_idxs[i],
                                                seq = seq,
                                                r =  first_rewards[i]
                                          } )
                    table.insert(beam, cand)
                end
                i = i + 1
            end
    
            -- beam search
            local best = nil
            local base_rewards = torch.CudaTensor(width):zero()
            local input = torch.CudaTensor(1, width)
            local steps = 0
            while steps < max_steps do
                steps = steps + 1

                -- prepare state
                local cur_state = { 
                                    torch.CudaTensor(1, width, state_vec_len),
                                    torch.CudaTensor(1, width, state_vec_len)
                                  }
                local cwss = {}
                for i = 1, width do
                    cur_state[1][1][i] = beam[i].state[1]
                    cur_state[2][1][i] = beam[i].state[2]
                    input[1][i] = beam[i].next_token
                    base_rewards[i] = beam[i].r
                end

                -- step RNN once
                local inter = model:forward({{cur_state}, input})
                local tok_rewards, tok_idxs = dec:topknext(inter,
                                                        math.max(width*(width+1), width*2+1),
                                                        input,
                                                        base_rewards)
                local nu_hidden = rnn:getLastHidden()[1]

                -- update beam
                local nu_beam = {}
                local i = 0
                while #nu_beam < width do
                    i = i + 1
                    local cur_r, n, w = tok_rewards[i], tok_idxs[i][1], tok_idxs[i][2]
                    local cand = beam[n]
                    local nu_seq = shallowcopy(cand.seq)
                    table.insert(nu_seq, w)
                    local nu_r = cur_r 
                    nu_r = nu_r + (r * math.pow(g, #nu_seq-init_len))
                    local nu_state = { 
                                       nu_hidden[1][1][n]:clone(),
                                       nu_hidden[2][1][n]:clone()
                                     }
                    
                    if w == term then
                        if best == nil or nu_r > best.r then
                            for i = 2, suffix:size(1) do
                                table.insert(nu_seq, suffix[i])
                            end
                            best = Candidate( 
                                              { 
                                                r = nu_r, 
                                                seq = nu_seq,
                                                state = nu_state
                                              }
                                            )
                        end
                    else
                        local nu_cand = Candidate(
                                                    {
                                                        r = nu_r,
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
            result = best.seq
            prefix = torch.CudaTensor(best.seq)
            state = { best.state }
        end
            
        return result
    end

decoders.stochastic_template_lengthbiased_beamsearch = 
    function(model, rnn, dec, width, template, dic, r, g, v)
        local prefix = template[1]
        local inter
        local init_len = template[1]:size(1)
        for c = 1, (#template-1) do
            -- prepare beam
            local state = rnn:initializeHidden(1)
            local state_vec_len = state[1][1]:size(3)
            local proc_prefix = prefix:view(prefix:size(1), 1)
            local inter = model:forward({state, proc_prefix})

            local base_probs = torch.CudaTensor(1):zero()
            local first_probs, first_idxs = dec:sampleknext(inter, 
                                                         width+1,
                                                         proc_prefix,
                                                         base_probs)
            first_idxs = first_idxs:t()
            first_idxs = first_idxs[2]
            local raw_hidden = rnn:getLastHidden()
            local suffix = template[c+1]
            local term = suffix[1]
            local init_hidden = { 
                                  raw_hidden[1][1]:clone(),
                                  raw_hidden[1][2]:clone()
                                }
            local init_seq = {}
            for i = 1, prefix:size(1) do table.insert(init_seq, prefix[i]) end
            local beam = {}
            local i = 1
            while #beam < width do 
                local w = first_idxs[i]
                if w ~= term then
                    local seq = shallowcopy(init_seq)
                    table.insert(seq, w)
                    local cand = Candidate( {
                                                state = init_hidden,
                                                next_token = first_idxs[i],
                                                seq = seq,
                                                r = first_probs[i] + r,
                                                p = first_probs[i]
                                          } )
                    table.insert(beam, cand)
                end
                i = i + 1
            end
    
            -- beam search
            local best = nil
            local base_probs = torch.CudaTensor(width):zero()
            local input = torch.CudaTensor(1, width)
            local steps = 0
            local max_r = r / (1 - g) 
            while (best == nil) or (best.r < (beam[1].p + max_r)) do
                steps = steps + 1
                if v > 0 then
                    for i = 1, #beam[1].seq do
                        io.write(dic.idx2word[beam[1].seq[i]] .. ' ')
                    end
                    print('')
                end

                -- prepare state
                local cur_state = { 
                                    torch.CudaTensor(1, width, state_vec_len),
                                    torch.CudaTensor(1, width, state_vec_len)
                                  }
                for i = 1, width do
                    cur_state[1][1][i] = beam[i].state[1]
                    cur_state[2][1][i] = beam[i].state[2]
                    input[1][i] = beam[i].next_token
                    base_probs[i] = beam[i].p
                end

                -- step RNN once
                local inter = model:forward({{cur_state}, input})
                local tok_probs, tok_idxs = dec:sampleknext(
                                                inter,
                                                width+1,
                                                input,
                                                base_probs)
                local nu_hidden = rnn:getLastHidden()[1]

                -- update beam
                local nu_beam = {}
                local i = 0
                while #nu_beam < width do
                    i = i + 1
                    local cur_p, n, w = tok_probs[i], tok_idxs[i][1], tok_idxs[i][2]
                    local cand = beam[n]
                    local nu_seq = shallowcopy(cand.seq)
                    table.insert(nu_seq, w)
                    local nu_r = cur_p + (r * math.pow(g, #nu_seq-init_len))
                    local nu_state = { 
                                       nu_hidden[1][1][n]:clone(),
                                       nu_hidden[2][1][n]:clone()
                                     }
                    
                    if w == term then
                        if best == nil or nu_r > best.r then
                            for i = 2, suffix:size(1) do
                                table.insert(nu_seq, suffix[i])
                            end
                            best = Candidate( 
                                              { 
                                                p = cur_p,
                                                r = nu_r, 
                                                seq = nu_seq,
                                                state = nu_state
                                              }
                                            )
                        end
                    else
                        local nu_cand = Candidate(
                                                    {
                                                        p = cur_p,
                                                        r = nu_r,
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
            result = best.seq
            prefix = torch.CudaTensor(best.seq)
            state = { best.state }
        end
            
        return result
    end


return decoders
