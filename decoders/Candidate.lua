local Candidate = torch.class("Candidate")

function Candidate:__init(opts)
    self.p = opts and opts.p 
    self.p = self.p or 0

    self.r = opts.r or p

    self.seq = opts and opts.seq
    self.seq = self.seq or {}

    if opts.toks then
        self.toks = opts.toks
    else
        local toks = {}
        for i = 1, #self.seq do
            toks[self.seq[i]] = true
        end
        self.toks = toks
    end

    self.state = opts and opts.state

    self.next_token = opts and opts.next_token

    self.t_loc = opts and opts.t_loc
    if not self.t_loc then
        self.t_loc = { chunk_idx=1, word_idx=1, looking_for=0 }
    end
end
