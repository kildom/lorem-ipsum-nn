import { useState, useEffect, useRef, Component, createRef } from 'react'
import AppBar from '@mui/material/AppBar';
import Box from '@mui/material/Box';
import ButtonGroup from '@mui/material/ButtonGroup';
import Toolbar from '@mui/material/Toolbar';
import Typography from '@mui/material/Typography';
import Button from '@mui/material/Button';
import IconButton from '@mui/material/IconButton';
import InputLabel from '@mui/material/InputLabel';
import LinearProgress from '@mui/material/LinearProgress';
import Alert from '@mui/material/Alert';
import Dialog from '@mui/material/Dialog';
import DialogActions from '@mui/material/DialogActions';
import DialogContent from '@mui/material/DialogContent';
import DialogContentText from '@mui/material/DialogContentText';
import DialogTitle from '@mui/material/DialogTitle';
import AlertTitle from '@mui/material/AlertTitle';
import TextField from '@mui/material/TextField';
import FormControlLabel from '@mui/material/FormControlLabel';
import Switch from '@mui/material/Switch';
import Stack from '@mui/material/Stack';
import Autocomplete from '@mui/material/Autocomplete';
import Select, { SelectChangeEvent } from '@mui/material/Select';
import Grid from '@mui/material/Grid';
import MenuItem from '@mui/material/MenuItem';
import Accordion from '@mui/material/Accordion';
import FormControl from '@mui/material/FormControl';
import AccordionActions from '@mui/material/AccordionActions';
import AccordionSummary from '@mui/material/AccordionSummary';
import AccordionDetails from '@mui/material/AccordionDetails';
import Paper from '@mui/material/Paper';
import HistoryEdu from '@mui/icons-material/HistoryEdu';
import GitHub from '@mui/icons-material/GitHub';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import Code from '@mui/icons-material/Code';
import './App.css'
import { LoremIpsum, ILoremIpsumOptions } from '../src/ts/lorem-ipsum';


interface ILoremIpsumInBg {
    done: boolean;
    canceled: boolean;
    progress: number;
    cancel: () => void;
}


function loremIpsumInBg(options: ILoremIpsumOptions, length: number, context: string, update: (text: string[] | string) => void): ILoremIpsumInBg {
    let cancel = false;
    let obj: ILoremIpsumInBg = {
        done: false,
        canceled: false,
        progress: 0,
        cancel: () => {
            obj.canceled = true;
            cancel = true;
        }
    }
    async function inner(): Promise<void> {
        await new Promise(resolve => setTimeout(resolve, 1));
        let ipsum = new LoremIpsum(options);
        let remaining = length - 1;
        if (context.length) {
            ipsum.setContext(context);
            remaining -= context.length;
            if (remaining < 2) {
                remaining = 2;
            }
        }
        while (remaining >= 0 && !cancel) {
            let endTime = Date.now() + 50;
            let updateText: string[] | string;
            if (options.paragraphs) {
                updateText = [];
                while (remaining >= 0 && !cancel && Date.now() < endTime) {
                    let paragraph = context;
                    context = '';
                    while (remaining >= 0) {
                        let letter = ipsum.next(remaining);
                        paragraph += letter;
                        remaining--;
                        if (letter === '\n') {
                            break;
                        }
                    }
                    updateText.push(paragraph);
                }
            } else {
                updateText = context;
                context = '';
                while (remaining >= 0) {
                    let letter = ipsum.next(remaining);
                    updateText += letter;
                    remaining--;
                    if (letter === '.' && (Date.now() >= endTime || cancel)) {
                        break;
                    }
                }
            }
            await new Promise(resolve => setTimeout(resolve, 2));
            obj.progress = (length - remaining) / length;
            obj.done = remaining < 0 || cancel;
            update(updateText);
        }
    }
    inner();
    return obj;
}

const countOptions = ['100', '200', '500', '1K', '2K', '5K', '10K', '20K', '50K', '100K', '200K', '500K', '1M'];
const heatOptions = ['12%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%', '100%', '120%', '140%', '160%', '180%', '200%'];

enum GeneratingState {
    NotStarted = 'NotStarted',
    Generating = 'Generating',
    Finished = 'Finished',
};

enum CopyState {
    None = 'None',
    Copied = 'Copied',
    Error = 'Error',
}

enum OutputMode {
    Formatted = 'Formatted',
    PlainText = 'PlainText',
    HTMLSource = 'HTMLSource',
}

enum ParagraphStyle {
    None = 'None',
    Small = 'Small',
    Normal = 'Normal',
    Large = 'Large',
}

interface IState {
    generatingState: GeneratingState;
    copyState: CopyState;
    optionsExpanded: boolean;
    startWithLoremIpsum: boolean;
    progress: number;
    count: string;
    seed: string;
    heat: string;
    language: string;
    paragraphs: ParagraphStyle;
    outputMode: OutputMode;
    visibleOutputMode: OutputMode;
    devDialogOpen: boolean;
};

const initialState: IState = {
    generatingState: GeneratingState.NotStarted,
    copyState: CopyState.None,
    optionsExpanded: false,
    startWithLoremIpsum: true,
    progress: 0,
    count: '10K',
    seed: 'Random',
    heat: '60%',
    language: 'la',
    paragraphs: ParagraphStyle.Normal,
    outputMode: OutputMode.Formatted,
    visibleOutputMode: OutputMode.Formatted,
    devDialogOpen: false,
};

type StateRef = React.RefObject<{
    state: IState;
    setState: React.Dispatch<React.SetStateAction<IState>>;
}>;

function parseCount(count: string): number {
    count = count.trim().toLowerCase();
    let multiplier = 1;
    if (count.endsWith('k')) multiplier = 1000;
    else if (count.endsWith('m')) multiplier = 1000000;
    let val = parseFloat(count.replace(/k|m/g, '').trim());
    if (isNaN(val)) return 10000;
    val *= multiplier;
    val = Math.round(val);
    if (val < 1) val = 1;
    if (val > 100000000) val = 100000000;
    return val;
}

function normalizeCount(count: string): string {
    let val = parseCount(count);
    if (val % 1000000 === 0) return Math.round(val / 1000000).toString() + 'M';
    if (val % 1000 === 0) return Math.round(val / 1000).toString() + 'K';
    return val.toString();
}

function parseHeat(heat: string): number {
    heat = heat.replace(/%/g, '').trim();
    if (heat === '') return 60;
    let val = parseInt(heat);
    if (isNaN(val)) return 60;
    if (val < 12) val = 12;
    if (val > 1600) val = 1600;
    return val;
}

function normalizeHeat(heat: string): string {
    return parseHeat(heat).toString() + '%';
}

function parseSeed(seed: string): number | 'Random' {
    seed = seed.trim();
    if (seed === '' || seed.toLowerCase().startsWith('r')) {
        return 'Random';
    }
    let val = parseInt(seed);
    if (isNaN(val) || val < 0 || val > 4294967295) return 'Random';
    return val;
}

function normalizeSeed(seed: string): string {
    return parseSeed(seed).toString();
}

class App extends Component<{}, IState> {

    loremIpsumState: ILoremIpsumInBg | undefined = undefined;
    state: IState;
    resultContentNodes: Node[];
    resultContainerRef: React.RefObject<HTMLDivElement | null>;

    constructor(props: {}) {
        super(props);
        this.state = initialState;
        this.resultContentNodes = [];
        this.resultContainerRef = createRef<HTMLDivElement>();
    }

    private refreshResultContainer() {
        if (this.resultContentNodes.length > 0) {
            let div = this.resultContainerRef.current;
            if (div && div.childNodes.length === 0) {
                for (let child of this.resultContentNodes as Node[]) {
                    div.appendChild(child);
                }
            }
        }
    }

    componentDidMount() {
        this.refreshResultContainer();
    }

    private appendNode(node: Node) {
        this.resultContentNodes.push(node);
        this.resultContainerRef.current?.appendChild(node);
    }

    private handleGenerate() {
        let outputMode = this.state.outputMode;
        let seed = parseSeed(this.state.seed);
        let count = parseCount(this.state.count);
        let options: ILoremIpsumOptions = {
            language: this.state.language,
            seed: seed === 'Random' ? undefined : seed,
            heat: parseHeat(this.state.heat) / 100,
        };
        switch (this.state.paragraphs) {
            case ParagraphStyle.Small:
                options.paragraphs = {
                    shorterVariance: 0.5,
                    mean: 3,
                    longerVariance: 1,
                }
                break;
            case ParagraphStyle.Normal:
                options.paragraphs = {
                    shorterVariance: 2,
                    mean: 5,
                    longerVariance: 4,
                }
                break;
            case ParagraphStyle.Large:
                options.paragraphs = {
                    shorterVariance: 4,
                    mean: 12,
                    longerVariance: 8,
                }
                break;
            case ParagraphStyle.None:
            default:
                break;
        }
        this.resultContentNodes = [];
        this.resultContainerRef.current?.replaceChildren();
        if (outputMode === OutputMode.HTMLSource) {
            this.appendNode(document.createTextNode('<p>'));
        }
        let context = '';
        if (this.state.startWithLoremIpsum) {
            context = 'Lorem ipsum dolor sit amet, ';
        }
        this.loremIpsumState = loremIpsumInBg(options, count, context, (text: string[] | string) => {
            this.refreshResultContainer();
            if (typeof text === 'string') {
                this.appendNode(document.createTextNode(text));
            } else {
                if (outputMode === OutputMode.Formatted) {
                    for (let paragraph of text) {
                        let p = document.createElement('p');
                        p.appendChild(document.createTextNode(paragraph.trim()));
                        this.appendNode(p);
                    }
                } else {
                    let concatenated = text.join('');
                    if (outputMode === OutputMode.PlainText) {
                        concatenated = concatenated.replace(/\n/g, '\n\n');
                    } else if (outputMode === OutputMode.HTMLSource) {
                        concatenated = concatenated.replace(/\n/g, '</p>\n\n<p>');
                    }
                    this.appendNode(document.createTextNode(concatenated));
                }
            }
            this.setState(prev => ({ ...prev, progress: this.loremIpsumState?.progress || 1 }));
            if (this.loremIpsumState?.done) {
                if (outputMode === OutputMode.HTMLSource) {
                    this.appendNode(document.createTextNode('</p>\n'));
                }
                this.setState(prev => ({ ...prev, generatingState: GeneratingState.Finished }));
                this.loremIpsumState = undefined;
            }
        });
        this.setState(prev => ({ ...prev, generatingState: GeneratingState.Generating, copyState: CopyState.None, progress: 0, visibleOutputMode: outputMode }));
    }

    private handleCancel() {
        this.loremIpsumState?.cancel();
    }

    private handleCopy() {
        try {
            const div = this.resultContainerRef.current;
            if (div) {
                if (this.state.visibleOutputMode !== OutputMode.Formatted) {
                    navigator.clipboard.writeText(div.innerText);
                } else {
                    const blob = new Blob([div.innerHTML], { type: 'text/html' });
                    const data = [new ClipboardItem({
                        'text/html': blob,
                        'text/plain': new Blob([div.innerText], { type: 'text/plain' })
                    })];
                    navigator.clipboard.write(data);
                }
            }
            this.setState(prev => ({ ...prev, copyState: CopyState.Copied }));
        } catch (e) {
            this.setState(prev => ({ ...prev, copyState: CopyState.Error }));
        }
    }

    render() {
        let state = this.state;
        return (<>

            <AppBar position="static">
                <Toolbar>
                    <IconButton size="large" edge="start" color="inherit" sx={{ mr: 2 }}><HistoryEdu /></IconButton>
                    <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
                        Lorem ipsum
                        <Typography variant="subtitle2" component="div" sx={{ flexGrow: 4 }}>
                            Deterministic Lorem Ipsum generator powered by a minimal neural network
                        </Typography>
                    </Typography>
                    &nbsp;&nbsp;&nbsp;
                    <Button color="success" startIcon={<Code />} variant='contained' onClick={() => this.setState(prev => ({ ...prev, devDialogOpen: true }))}>Developers</Button>
                    &nbsp;&nbsp;&nbsp;
                    <Button color="success" startIcon={<GitHub />} variant='contained' href='https://github.com/kildom/lorem-ipsum-nn/'>GitHub</Button>
                </Toolbar>
            </AppBar>

            <Accordion style={{ maxWidth: '840px', margin: '30px auto' }} elevation={3}
                onChange={(e, expanded) => this.setState(prev => ({ ...prev, optionsExpanded: expanded }))}
                expanded={state.optionsExpanded}
            >
                <AccordionSummary
                    expandIcon={<ExpandMoreIcon />}
                    aria-controls="panel1-content"
                    id="panel1-header"
                >
                    <Typography variant='h6'>Options</Typography>
                </AccordionSummary>
                <AccordionDetails>
                    <Grid container spacing={2}>
                        <Grid size={3}>
                            <Autocomplete fullWidth
                                onBlur={e => this.setState(prev => ({ ...prev, count: normalizeCount(prev.count) }))}
                                value={state.count}
                                inputValue={state.count}
                                onInputChange={(e, newValue) => this.setState(prev => ({ ...prev, count: newValue }))}
                                freeSolo
                                options={countOptions}
                                filterOptions={(x) => x}
                                renderInput={(params) => (
                                    <TextField {...params}
                                        label="Number of characters"
                                        fullWidth
                                    />
                                )}
                            />
                        </Grid>
                        <Grid size={3}>
                            <FormControl fullWidth>
                                <InputLabel id="demo-simple-select-label">Language stylization</InputLabel>
                                <Select
                                    value={state.language}
                                    onChange={e => this.setState(prev => ({ ...prev, language: e.target.value }))}
                                    labelId="demo-simple-select-label"
                                    id="demo-simple-select"
                                    label="Language stylization"
                                >
                                    {
                                        Object.entries(LoremIpsum.languages()).map(([code, name]) => (
                                            <MenuItem key={code} value={code}>{code} ({name})</MenuItem>
                                        ))
                                    }
                                </Select>
                            </FormControl>
                        </Grid>
                        <Grid size={3}>
                            <Autocomplete fullWidth
                                onBlur={e => this.setState(prev => ({ ...prev, seed: normalizeSeed(prev.seed) }))}
                                value={state.seed}
                                inputValue={state.seed}
                                onInputChange={(e, newValue) => this.setState(prev => ({ ...prev, seed: newValue }))}
                                freeSolo
                                options={['Random']}
                                filterOptions={(x) => x}
                                renderInput={(params) => (
                                    <TextField {...params}
                                        label="Seed"
                                        fullWidth
                                    />
                                )}
                            />

                        </Grid>
                        <Grid size={3}>
                            <Autocomplete fullWidth
                                onBlur={e => this.setState(prev => ({ ...prev, heat: normalizeHeat(prev.heat) }))}
                                value={state.heat}
                                inputValue={state.heat}
                                onInputChange={(e, newValue) => this.setState(prev => ({ ...prev, heat: newValue }))}
                                freeSolo
                                options={heatOptions}
                                filterOptions={(x) => x}
                                renderInput={(params) => (
                                    <TextField {...params}
                                        label="Heat"
                                        fullWidth
                                    />
                                )}
                            />
                        </Grid>

                        <Grid size={3}>
                            <FormControl fullWidth>
                                <InputLabel id="format-label">Output format</InputLabel>
                                <Select
                                    value={state.outputMode}
                                    onChange={e => this.setState(prev => ({ ...prev, outputMode: e.target.value }))}
                                    labelId="format-label"
                                    label="Output format"
                                >
                                    <MenuItem value={OutputMode.Formatted}>Formatted</MenuItem>
                                    <MenuItem value={OutputMode.PlainText}>Plain text</MenuItem>
                                    <MenuItem value={OutputMode.HTMLSource}>HTML source code</MenuItem>
                                </Select>
                            </FormControl>
                        </Grid>

                        <Grid size={3}>
                            <FormControl fullWidth>
                                <InputLabel id="format-label">Paragraphs</InputLabel>
                                <Select
                                    value={state.paragraphs}
                                    onChange={e => this.setState(prev => ({ ...prev, paragraphs: e.target.value }))}
                                    labelId="format-label"
                                    label="Paragraphs"
                                >
                                    <MenuItem value={ParagraphStyle.None}>None</MenuItem>
                                    <MenuItem value={ParagraphStyle.Small}>Small</MenuItem>
                                    <MenuItem value={ParagraphStyle.Normal}>Normal</MenuItem>
                                    <MenuItem value={ParagraphStyle.Large}>Large</MenuItem>
                                </Select>
                            </FormControl>
                        </Grid>
                        <Grid size={6} sx={{ display: 'flex', alignItems: 'center' }}>
                            <FormControlLabel control={<Switch
                                size='medium'
                                checked={state.startWithLoremIpsum}
                                onChange={e => this.setState(prev => ({ ...prev, startWithLoremIpsum: e.target.checked }))}
                            />} label='Start with "Lorem ipsum"' />
                        </Grid>

                    </Grid>
                </AccordionDetails>
            </Accordion>

            <Paper elevation={3} style={{ maxWidth: '800px', margin: '30px auto', padding: '20px' }}>
                {(state.generatingState === GeneratingState.NotStarted) &&
                    <div style={{ textAlign: 'center' }}>
                        <Button variant='contained' color='success' size='large' style={{ width: 250 }} onClick={() => this.handleGenerate()}>Generate</Button>
                    </div>
                }
                {(state.generatingState === GeneratingState.Generating) &&
                    <>
                        <div style={{ textAlign: 'center' }}>
                            <Button variant='contained' color='error' size='large' style={{ width: 250 }} onClick={() => this.handleCancel()}>Cancel</Button>
                        </div>
                        <LinearProgress variant="determinate" value={state.progress * 100} style={{ marginTop: 20 }} />
                    </>
                }
                {(state.generatingState === GeneratingState.Finished) &&
                    <div style={{ textAlign: 'center' }}>
                        <Button variant='contained' color='success' size='large' style={{ width: 200 }} onClick={() => this.handleGenerate()}>Regenerate</Button>
                        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
                        <Button variant='contained' color='primary' size='large' style={{ width: 200 }} onClick={() => this.handleCopy()}>Copy to clipboard</Button>
                    </div>
                }

                {(state.copyState === CopyState.Copied) &&
                    <Alert severity="success" style={{ cursor: 'pointer', marginTop: 20 }} onClick={() => this.setState(prev => ({ ...prev, copyState: CopyState.None }))}>
                        <AlertTitle>Success</AlertTitle>
                        The text was successfully copied to the clipboard.
                    </Alert>
                }

                {(state.copyState === CopyState.Error) &&
                    <Alert severity="error" style={{ cursor: 'pointer', marginTop: 20 }} onClick={() => this.setState(prev => ({ ...prev, copyState: CopyState.None }))}>
                        <AlertTitle>Error</AlertTitle>
                        There was an error copying the text to the clipboard. Try to manually select the text and copy it.
                    </Alert>
                }

            </Paper>

            <Paper elevation={3} style={{ maxWidth: '800px', margin: '30px auto 50px', padding: '20px', visibility: state.generatingState === GeneratingState.NotStarted ? 'hidden' : 'visible' }}>
                <div
                    className={this.state.visibleOutputMode === OutputMode.Formatted ? 'result-formatted' : 'result-plain'}
                    ref={this.resultContainerRef}
                ></div>
            </Paper>
            <Dialog
                open={state.devDialogOpen}
                onClose={() => this.setState(prev => ({ ...prev, devDialogOpen: false }))}
                aria-labelledby="alert-dialog-title"
                aria-describedby="alert-dialog-description"
            >
                <DialogTitle id="alert-dialog-title">
                    Are you a developer and want to use this generator?
                </DialogTitle>
                <DialogContent>
                    <DialogContentText id="alert-dialog-description">
                        The generator is available on multiple platforms.
                        You can use it in your projects as a library or as a standalone application.
                        <ul>
                            <li><b>JavaScript package</b> - available on <a href="https://www.npmjs.com/package/lorem-ipsum-nn">npm</a></li>
                            <li><b>Python package</b> - available on <a href="https://pypi.org/project/lorem-ipsum-nn/">PyPI</a></li>
                            <li><b>C/C++ library</b> - available on GitHub as a <a href="https://github.com/kildom/lorem-ipsum-nn/tree/main/src/c_lib">source code</a></li>
                            <li><b>CLI tool</b> - downloadable from <a href="https://github.com/kildom/lorem-ipsum-nn/releases">GitHub releases</a></li>
                        </ul>
                    </DialogContentText>
                </DialogContent>
                <DialogActions>
                    <Button variant='outlined' onClick={() => this.setState(prev => ({ ...prev, devDialogOpen: false }))} autoFocus style={{minWidth: 180}}>OK, Great!</Button>
                </DialogActions>
            </Dialog>

        </>)
    }
}

export default App
