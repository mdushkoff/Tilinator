--[[
    This includes various routines for tiling images
    in order to create seamless tiling textures.
    This requires the image package to be installed.

    Author: Michael Dushkoff (mad1841@rit.edu)
    Date: 12/14/2016
--]]

-- Requirements
require 'image'

-- Global class definition
Tile = {}

--[[
    
--]]
function Tile.simple(img,params)
    -- Extract parameters
    local maskSize = params.maskSize or 1.0
    local maskBlur = params.maskBlur or 0.5
    local clusterWidth = params.clusterWidth or torch.floor(img:size(img:dim())/10)
    local clusterHeight = params.clusterHeight or torch.floor(img:size(img:dim()-1)/10)
    local sizeVar = params.sizeVar or 0.0
    local overlap = params.overlap or 0.25
    local rotation = params.rotation or 0.0
    local rotationVar = params.rotationVar or 0.0
    local bgColor = params.bgColor or {0,0,0}
    local octave = params.octave or 2
    local seed = params.seed or 0

    -- Assertions (TODO)
    

    -- Compute constants
    local sz = torch.totable(#img) -- Tile size
    local h = sz[#sz-1]  -- Height
    local w = sz[#sz]    -- Width
    local x1 = math.floor((w/2)-(clusterWidth/2))
    local x2 = x1+clusterWidth
    local y1 = math.floor((h/2)-(clusterHeight/2))
    local y2 = y1+clusterHeight
    sz[#sz-1] = clusterHeight
    sz[#sz] = clusterWidth
    local repeats = 2^octave
    local rsz = torch.totable(#img) -- Resize
    rsz[#sz-1] = math.floor(rsz[#sz-1]/(repeats*(1-overlap)))  -- FIXME (include aspect ratio)
    rsz[#sz] = math.floor(rsz[#sz]/(repeats*(1-overlap)))      -- FIXME (include aspect ratio)
    local ox = math.floor(rsz[#sz]*overlap/2)                  -- FIXME (include aspect ratio)
    local oy = math.floor(rsz[#sz-1]*overlap/2)                -- FIXME (include aspect ratio)

    -- Preallocate output image
    local out = torch.Tensor(#img):zero()

    --[[ Compute tiled image output ]]--
    -- Create small tile to sample from and its mask
    --local tile = torch.Tensor(table.unpack(sz))
    --local mask = image.gaussian{height=sz[#sz-1],width=sz[#sz],sigma=maskBlur,amplitude=1.0}
    local mask = image.gaussian{height=rsz[#sz-1],width=rsz[#sz],sigma=maskBlur,amplitude=1.0}
    mask = mask:reshape(1,mask:size(1),mask:size(2))
    mask = mask:repeatTensor(rsz[1],1,1)
    local tile = image.scale(mask,image.crop(img,x1,y1,x2,y2))
    local tW = tile:size(tile:dim())
    local tH = tile:size(tile:dim()-1)
    local accumMask = torch.Tensor(#img):fill(0)  -- Accumulator for mask values

    -- Create rotation and scale constants
    local rot = torch.Tensor(1,repeats,repeats):fill(rotation)  -- FIXME: Add variation
    local scl = torch.Tensor(#rot):fill(1)  -- FIXME: Add variation

    -- Create variables for placement
    local xi1 = 1
    local xi2 = 1
    local yi1 = 1
    local yi2 = 1

    -- Perform tiling
    for y=1,repeats do
        -- y-dimension crop assignment
        if (y==1) then
            y1 = oy-1
            y2 = rsz[#rsz-1]
            yi1 = 1
            yi2 = yi1+(rsz[#rsz-1]-oy)
        elseif (y==repeats) then
            y1 = 0
            y2 = rsz[#rsz-1]-oy
            if (rsz[#rsz-1] % 2 ~= 0) then y2 = y2-1; end -- Parity correction
            yi1 = (y-1)*math.floor((w/repeats))+1-(oy)
            yi2 = w
        else
            y1 = 0
            y2 = rsz[#rsz-1]
            --yi1 = (y-1)*math.floor((h/repeats))+1-(oy*2)
            yi1 = (y-1)*math.floor((h/repeats))+1-(oy)
            --if (rsz[2] % 2 ~= 0) then yi1 = yi1+1 end -- Correct for parity
            yi2 = yi1+rsz[#rsz-1]-1
        end
        for x=1,repeats do
            -- X-dimension crop assignment
            if (x==1) then
                x1 = ox-1
                x2 = rsz[#rsz]
                xi1 = 1
                xi2 = xi1+(rsz[#rsz]-ox)
            elseif (x==repeats) then
                x1 = 0
                x2 = rsz[#rsz]-ox
                if (rsz[#rsz] % 2 ~= 0) then x2 = x2-1; end -- Parity correction
                xi1 = (x-1)*math.floor((w/repeats))+1-(ox)
                xi2 = w
            else
                x1 = 0
                x2 = rsz[#rsz]
                xi1 = (x-1)*math.floor((w/repeats))+1-(ox)
                xi2 = xi1+rsz[#rsz]-1
            end
            
            -- Place initial tile
            out[{{},{yi1,yi2},{xi1,xi2}}]:add( torch.cmul(image.crop(tile,x1,y1,x2,y2),image.crop(mask,x1,y1,x2,y2)) )

            -- Accumulate mask values
            accumMask[{{},{yi1,yi2},{xi1,xi2}}]:add(image.crop(mask,x1,y1,x2,y2))

            -- Place wrapped cuts TODO
            
        end
    end

    -- Divide by mask values
    out:cdiv(accumMask)

    return out
end

--[[

--]]
function Tile.smooth(img,params)
    -- Extract parameters
    local maskSize = params.maskSize or 1.0
    local maskBlur = params.maskBlur or 0.5
    local clusterWidth = params.clusterWidth or torch.floor(img:size(img:dim())/10)
    local clusterHeight = params.clusterHeight or torch.floor(img:size(img:dim()-1)/10)
    local sizeVar = params.sizeVar or 0.0
    local rotation = params.rotation or 0.0
    local rotationVar = params.rotationVar or 0.0
    local bgColor = params.bgColor or {0,0,0}
    local octave = params.octave or 2
    local seed = params.seed or 0

    -- Assertions (TODO)
    

    -- Calculate constants
    local sz = torch.totable(#img)
    local h = sz[#sz-1]
    local w = sz[#sz]
    local repeats = 2^octave
    local sx = w/(repeats*clusterWidth)   -- Scale factor (X-Dimension)
    local sy = h/(repeats*clusterHeight)  -- Scale factor (Y-Dimension)
    local tsz = torch.totable(#img) -- Tile resize dimensions
    tsz[#tsz-1] = math.floor(h*sy)
    tsz[#tsz] = math.floor(w*sx)
    local ox = math.floor((tsz[#tsz]-w/repeats)/2) -- Outer LHS (X-Dimension)
    local oy = math.floor((tsz[#tsz-1]-h/repeats)/2) -- Outer THS (Y-Dimension)
    local osz = torch.totable(#img)
    osz[#osz-1] = h+oy*2
    osz[#osz] = w+ox*2

    -- Parity correction for output size
    if (tsz[#tsz-1] % 2 ~= 0) then osz[#osz-1] = osz[#osz-1] + 1; end
    if (tsz[#tsz] % 2 ~= 0) then osz[#osz] = osz[#osz] + 1; end 

    -- Allocate space for the base output and mask accumulator
    local out = torch.Tensor(table.unpack(osz))
    local accMask = torch.Tensor(#out):zero()

    -- Calculate the mask
    local mask = image.gaussian{height=tsz[#tsz-1],width=tsz[#tsz],sigma=maskBlur,amplitude=1.0}
    mask = mask:reshape(1,mask:size(1),mask:size(2))
    mask = mask:repeatTensor(tsz[1],1,1)

    -- Create tile (Pre-computed with mask)
    local tile = image.scale(img,tsz[#tsz],tsz[#tsz-1]):cmul(mask)

    -- Extend colorspace to match total size of first dimension
    for i=4,sz[1] do bgColor[i] = 1 end

    -- Tile and accumulate
    local xi1 = 1
    local xi2 = 1
    local yi1 = 1
    local yi2 = 1
    for y=1,repeats do
        yi1 = (y-1)*(h/repeats)+1
        yi2 = yi1+tsz[#tsz-1]-1
        for x=1,repeats do
            xi1 = (x-1)*(w/repeats)+1
            xi2 = xi1+tsz[#tsz]-1
            out[{{},{yi1,yi2},{xi1,xi2}}]:add(tile)
            accMask[{{},{yi1,yi2},{xi1,xi2}}]:add(mask)
        end
    end

    --[[ Add edges ]]--
    -- Left -> Right
    out[{{},{oy+1,oy+h},{w+1,ox+w}}]:add(out[{{},{oy+1,oy+h},{1,ox}}])
    accMask[{{},{oy+1,oy+h},{w+1,ox+w}}]:add(accMask[{{},{oy+1,oy+h},{1,ox}}])
    -- Right -> Left
    out[{{},{oy+1,oy+h},{ox+1,ox+ox}}]:add(out[{{},{oy+1,oy+h},{ox+w+1,ox+w+ox}}])
    accMask[{{},{oy+1,oy+h},{ox+1,ox+ox}}]:add(accMask[{{},{oy+1,oy+h},{ox+w+1,ox+w+ox}}])
    -- Top -> Bottom
    out[{{},{h+1,oy+h},{ox+1,ox+w}}]:add(out[{{},{1,oy},{ox+1,ox+w}}])
    accMask[{{},{h+1,oy+h},{ox+1,ox+w}}]:add(accMask[{{},{1,oy},{ox+1,ox+w}}])
    -- Bottom -> Top
    out[{{},{oy+1,oy+oy},{ox+1,ox+w}}]:add(out[{{},{oy+h+1,oy+h+oy},{ox+1,ox+w}}])
    accMask[{{},{oy+1,oy+oy},{ox+1,ox+w}}]:add(accMask[{{},{oy+h+1,oy+h+oy},{ox+1,ox+w}}])

    -- Crop output and mask accumulator
    out = image.crop(out,ox,oy,ox+w,oy+h)
    accMask = image.crop(accMask,ox,oy,ox+w,oy+h)

    -- Divide the output by the accumulator mask (Excluding zeros)
    local ez = accMask:eq(0)
    local nez = accMask:ne(0)
    --out = out:cdiv(accMask):cmul(nez)+torch.Tensor(bgColor):reshape(sz[1],1,1):repeatTensor(1,sz[2],sz[3]):cmul(ez)
    out = out:cdiv(accMask) -- Reduce by mask factor
    local col = torch.Tensor(bgColor):reshape(sz[1],1,1):repeatTensor(1,sz[2],sz[3])
    out[ez] = col[ez]

    return out
end
