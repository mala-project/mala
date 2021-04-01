#!/bin/bash

rsync -rn --verbose --exclude pattern source destination


# Alternative
#rsync -avz --exclude=*temp --exclude=*out
