import { createTheme, PaletteMode } from '@mui/material';

export const getTheme = (mode: PaletteMode) => {
  return createTheme({
    palette: {
      mode,
      primary: {
        main: '#3f51b5',
        light: '#757de8',
        dark: '#002984',
        contrastText: '#fff',
      },
      secondary: {
        main: '#f50057',
        light: '#ff4081',
        dark: '#c51162',
        contrastText: '#fff',
      },
      background: {
        default: mode === 'light' ? '#f5f5f5' : '#121212',
        paper: mode === 'light' ? '#ffffff' : '#1e1e1e',
      },
    },
    typography: {
      fontFamily: '"Roboto", "Helvetica", "Arial", sans-serif',
      h1: {
        fontWeight: 700,
      },
      h2: {
        fontWeight: 600,
      },
      h3: {
        fontWeight: 600,
      },
      h4: {
        fontWeight: 600,
      },
      h5: {
        fontWeight: 500,
      },
      h6: {
        fontWeight: 500,
      },
    },
    shape: {
      borderRadius: 8,
    },
    components: {
      MuiButton: {
        styleOverrides: {
          root: {
            textTransform: 'none',
            fontWeight: 600,
          },
          contained: {
            boxShadow: '0px 3px 6px rgba(0, 0, 0, 0.1)',
          },
        },
      },
      MuiCard: {
        styleOverrides: {
          root: {
            borderRadius: 12,
            boxShadow: mode === 'light' 
              ? '0 4px 20px rgba(0, 0, 0, 0.08)'
              : '0 4px 20px rgba(0, 0, 0, 0.3)',
            transition: 'transform 0.2s, box-shadow 0.2s',
            '&:hover': {
              transform: 'translateY(-4px)',
              boxShadow: mode === 'light'
                ? '0 8px 25px rgba(0, 0, 0, 0.15)'
                : '0 8px 25px rgba(0, 0, 0, 0.5)',
            },
          },
        },
      },
      MuiPaper: {
        styleOverrides: {
          root: {
            transition: 'all 0.2s ease-in-out',
          },
        },
      },
      MuiChip: {
        styleOverrides: {
          root: {
            fontWeight: 500,
          },
        },
      },
      MuiTab: {
        styleOverrides: {
          root: {
            textTransform: 'none',
            fontWeight: 600,
            minWidth: 'auto',
          },
        },
      },
      MuiTableCell: {
        styleOverrides: {
          head: {
            fontWeight: 700,
            backgroundColor: mode === 'light' ? '#f5f7fa' : '#333',
          },
        },
      },
    },
  });
};

export default getTheme;